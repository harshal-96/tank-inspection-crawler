"""Isaac Sim welding station: run a campaign of parts and capture what the cameras see.

Run with Isaac's own interpreter, not the project venv:

    E:\\Issac-sim\\python.bat cell\\sim\\weld_station.py --trials 20

What one trial produces:
  - a part arriving under physics at a randomised offset, settling against the stop
  - the true joint position, computed from the settled pose, which is the ground
    truth the detector is scored against
  - a pre-weld scan image for the detector
  - a post-weld bead image carrying a deliberately injected defect, or none

Nothing here runs the detector or the cloud inspector. It writes a manifest and
the images; the detector pass and the inspection pass consume them afterwards.
They are kept separate because the graphics card cannot hold Isaac and an
inference engine at the same time.

Three renderer behaviours this is built around, all of them expensive to
rediscover:

  1. After the first composition, ordinary transform edits on scene geometry never
     reach the headless renderer. Only poses driven through the replicator graph
     and changes to camera transform values do. So physics is settled first, and
     every pose that must appear on camera goes through a replicator sequence.
  2. The writer lags the scene by one frame. Every capture is therefore two steps,
     and only the second is kept.
  3. Geometry marked with the guide purpose is solid to physics but excluded from
     the render, which is how the part is constrained on its way in without the
     hardware appearing in the photograph.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--defect-rate", type=float, default=0.35)
    parser.add_argument("--out", default="data/captures")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Physics only. Validates arrivals and ground truth without the renderer.",
    )
    parser.add_argument(
        "--no-post",
        action="store_true",
        help="Do not register parts with the service. Useful when testing offline.",
    )
    parser.add_argument("--api", default=None, help="Service base URL")
    parser.add_argument("--subframes", type=int, default=12)
    parser.add_argument(
        "--views",
        type=int,
        default=5,
        help="Candidate viewing angles per part. The loop picks among these.",
    )
    parser.add_argument(
        "--orbit-range",
        type=float,
        default=45.0,
        help="Widest swing either side of square, in degrees.",
    )
    parser.add_argument("--no-guides", action="store_true", help="Debug: omit guide geometry")
    parser.add_argument("--no-bead", action="store_true", help="Debug: omit bead geometry")
    parser.add_argument(
        "--xray",
        action="store_true",
        help="Debug: label every prim with its own name so the segmentation mask "
        "identifies whatever owns a given pixel.",
    )
    parser.add_argument(
        "--bounds",
        action="store_true",
        help="Print world bounding boxes of the scene and exit. Never trust a "
        "transform you have not measured.",
    )
    return parser.parse_args()


ARGS = parse_args()

# The simulation app has to exist before anything from omni or pxr is imported.
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleRigidPrim  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdSemantics,
    UsdShade,
)

from cell import defects as defect_lib  # noqa: E402
from cell import geometry as geo  # noqa: E402
from cell.client import CellClient  # noqa: E402

OUT_DIR = (PROJECT_ROOT / ARGS.out).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Where beads are laid out for the post-weld photographs. They are built once,
# well clear of the fixture, and the camera visits each in turn.
BEAD_FARM_X = -6.0
BEAD_FARM_PITCH = 4.0
PART_PARK = (-20.0, 0.0, 0.005)

RESOLUTION = (640, 640)
# Square, to match the square render product. A 4:3 aperture against a 1:1 target
# is how a subject the geometry says is centred ends up outside the frame.
APERTURE_MM = 20.955

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()


# --------------------------------------------------------------------------- scene


_MATERIALS: dict[tuple, object] = {}


def surface(color) -> object:
    """A real shaded material for a colour, cached.

    displayColor alone leaves a prim on the renderer's fallback path, where it can
    ignore the lighting entirely and draw black. A UsdPreviewSurface is what the
    path tracer actually wants.
    """
    key = tuple(round(c, 4) for c in color)
    if key in _MATERIALS:
        return _MATERIALS[key]

    name = "mat_%03d_%03d_%03d" % tuple(int(c * 255) for c in key)
    material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*key))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _MATERIALS[key] = material
    return material


def shade(prim, color) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(surface(color))


def box(path: str, size, translate, color=(0.55, 0.57, 0.6)):
    """An axis-aligned box.

    Built as a scaled cube. An explicit mesh was tried, to rule out normals going
    wrong under the sixty-to-one scales these panels use; it changed the shading
    not at all and cost the collisions, because a mesh cannot be a dynamic
    collider and PhysX silently filled the hollow part in with a convex hull.
    """
    prim = UsdGeom.Cube.Define(stage, path)
    # Leave size at its default of 2 and scale by half the wanted dimension, so
    # the cube's authored extent and its scale stay consistent.
    prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
    prim.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    shade(prim, color)
    return prim


def static_box(path, size, translate, color=(0.55, 0.57, 0.6), guide=False):
    prim = box(path, size, translate, color)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    if guide:
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set("guide")
    return prim


def build_fixture() -> None:
    """Table, end stop, locators and the guide rails that square the part up."""
    static_box("/World/Fix/Table", (1.6, 1.7, 0.08), (geo.FIXTURE_X, 0, geo.TABLE_TOP - 0.04))
    static_box(
        "/World/Fix/Stop",
        (0.05, 1.3, 0.12),
        (geo.FIXTURE_X + 0.485, 0, geo.TABLE_TOP + 0.06),
        color=(0.5, 0.2, 0.1),
    )
    # The approach slide, carrying the part from the conveyor hand-off to the
    # fixture. Without it the part spawns in mid air, falls the better part of a
    # metre and tumbles in. Its top sits half a millimetre proud of the table so
    # the trailing edge steps DOWN at the joint between them; flush or below and
    # the edge catches on the seam and the part stalls short of the stop.
    static_box("/World/Ground", (40.0, 40.0, 0.02), (0.0, 0.0, -0.01), color=(0.22, 0.22, 0.24))

    slide_top = geo.TABLE_TOP + 0.0005
    if ARGS.no_guides:
        return
    static_box(
        "/World/Fix/Slide",
        (3.20, 1.70, 0.08),
        (1.58, 0, slide_top - 0.04),
        guide=True,
    )
    # Rails constrain the part on the way in without appearing on camera.
    for index, side in ((0, -1), (1, 1)):
        static_box(
            "/World/Guide_%d" % index,
            (7.0, 0.04, 0.10),
            (0.8, side * 0.66, geo.TABLE_TOP + 0.05),
            guide=True,
        )


def slide_material():
    """A low-friction contact so the part glides to the stop instead of gripping.

    A conveyor hand-off is nearly frictionless compared with steel on a dry table.
    With default friction the part digs in, the push torques it about its centre,
    and it arrives rotated by tens of degrees, which no fixtured cell would allow.
    """
    material = UsdShade.Material.Define(stage, "/World/Physics/SlideMat")
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.06)
    physics_material.CreateDynamicFrictionAttr(0.05)
    return material


def build_part(path: str, physics: bool):
    """A closed sheet-metal cabinet: six panels enclosing a box.

    The weld joint is the vertical edge where the right panel meets the front
    panel, and it runs the full height of the cabinet.

    Every panel spans the full outer dimension in the two axes it is not thin in,
    so panels OVERLAP at every edge. Butting them to a shared plane leaves a
    zero-width seam that the renderer draws as a crack through to the unlit
    interior. Closing the box also removes the thin upward-facing rims of an open
    tray, which the renderer draws black however the scene is lit.
    """
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*PART_PARK))
    root.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))

    w, h, d, t = geo.PART_W, geo.PART_H, geo.PART_D, geo.PART_T
    panels = {
        "bottom": ((w, h, t), (0, 0, t / 2)),
        "top": ((w, h, t), (0, 0, d - t / 2)),
        "front": ((w, t, d), (0, -(h / 2 - t / 2), d / 2)),
        "back": ((w, t, d), (0, (h / 2 - t / 2), d / 2)),
        "left": ((t, h, d), (-(w / 2 - t / 2), 0, d / 2)),
        "right": ((t, h, d), ((w / 2 - t / 2), 0, d / 2)),
    }
    material = slide_material() if physics else None
    for name, (size, offset) in panels.items():
        panel = box(f"{path}/{name}", size, offset, color=(0.62, 0.64, 0.68))
        if physics:
            UsdPhysics.CollisionAPI.Apply(panel.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(panel.GetPrim()).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

    if physics:
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr().Set(38.0)
        # Without this the push torques the part and it arrives spinning.
        PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim()).CreateAngularDampingAttr(1.5)
    return root


def ellipsoid(path: str, radii, translate, color=(0.55, 0.57, 0.6)):
    """A sphere scaled into an ellipsoid, for weld ripples and pores."""
    prim = UsdGeom.Sphere.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
    prim.AddScaleOp().Set(Gf.Vec3f(*radii))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    shade(prim, color)
    return prim


def build_bead(
    path: str,
    plan,
    tag_semantics: bool = False,
    detailed: bool = False,
    seed: int = 0,
) -> None:
    """Weld bead for one part, built from its plan so the defect is real geometry.

    Two levels of detail, because the two cameras want different things. The wide
    scan sees the bead a metre away, where a ripple is smaller than a pixel, so it
    gets a smooth run of weld metal. The inspection close-up gets the real thing: a
    row of overlapping pools, which is what gives a weld its stack-of-dimes ripple
    and what a defect actually disturbs.

    The bead lies on the OUTSIDE of the corner, where a fillet weld sits. Centred on
    the joint it would be swallowed by the panels and photograph as a dark crevice.
    """
    UsdGeom.Xform.Define(stage, path)
    if ARGS.no_bead:
        return

    rng = random.Random(seed)
    local_x = geo.JOINT_LOCAL[0] + geo.BEAD_PROUD
    local_y = geo.JOINT_LOCAL[1] - geo.BEAD_PROUD
    bead_w, bead_d, full_len = geo.BEAD_SIZE
    base_z = geo.BEAD_Z - full_len / 2

    def along(fraction: float) -> float:
        return base_z + fraction * full_len

    tagged = []

    if not detailed:
        for index, (start, end) in enumerate(plan.segments):
            length = (end - start) * full_len
            if length <= 1e-4:
                continue
            centre = (start + end) / 2
            tagged.append(
                box(
                    f"{path}/bead_{index}",
                    (bead_w, bead_d, length),
                    (local_x, local_y, along(centre)),
                    color=(0.78, 0.68, 0.48),
                )
            )
    else:
        leg = geo.FILLET_LEG
        pitch = geo.RIPPLE_PITCH
        count = 0
        for start, end in plan.segments:
            span = (end - start) * full_len
            ripples = max(1, int(span / pitch))
            for i in range(ripples):
                fraction = start + (i + 0.5) * pitch / full_len
                if fraction > end:
                    break
                # Real weld pools vary. Perfectly uniform ripples read as a machined
                # part rather than a weld.
                swell = rng.uniform(0.86, 1.14)
                wander = rng.uniform(-0.0004, 0.0004)
                shade = rng.uniform(-0.05, 0.05)
                tagged.append(
                    ellipsoid(
                        f"{path}/ripple_{count}",
                        (leg / 2 * swell, leg / 2 * swell, pitch * 0.62),
                        (local_x + wander, local_y - wander, along(fraction)),
                        color=(0.72 + shade, 0.63 + shade, 0.45 + shade),
                    )
                )
                count += 1

        # Heat tint: the parent metal discolours in bands either side of the weld,
        # straw closest and blue-grey further out.
        for offset, tint in ((0.010, (0.68, 0.55, 0.30)), (0.019, (0.42, 0.44, 0.52))):
            box(
                f"{path}/tint_side_{int(offset * 1000)}",
                (0.0012, 0.008, full_len),
                (local_x + offset, local_y - 0.004, geo.BEAD_Z),
                color=tint,
            )
            box(
                f"{path}/tint_front_{int(offset * 1000)}",
                (0.008, 0.0012, full_len),
                (local_x - 0.004, local_y - offset, geo.BEAD_Z),
                color=tint,
            )

    if tag_semantics:
        for prim in tagged:
            UsdSemantics.LabelsAPI.Apply(prim.GetPrim(), "class").GetLabelsAttr().Set(
                ["joint_target"]
            )

    # ---------------------------------------------------------------- defects

    for index, blob in enumerate(plan.blobs):
        if blob.kind == defect_lib.POROSITY:
            # A pore is a cavity in the bead face, so it sits just inside the crown.
            centre = (
                local_x + blob.offset - 0.001,
                local_y - blob.offset + 0.001,
                along(blob.along),
            )
            colour = (0.06, 0.055, 0.05)
            radii = (blob.radius, blob.radius, blob.radius)
        else:
            # Spatter lands outboard, on the parent metal beside the weld.
            centre = (
                local_x + abs(blob.offset),
                local_y - abs(blob.offset) * 0.3,
                along(blob.along),
            )
            colour = (0.66, 0.60, 0.46)
            radii = (blob.radius, blob.radius, blob.radius * 0.7)
        ellipsoid(f"{path}/blob_{index}", radii, centre, color=colour)

    if plan.groove is not None:
        start, end, depth = plan.groove
        # Undercut is a groove melted into the parent metal along the bead toe,
        # so it runs beside the weld, not through it.
        box(
            f"{path}/undercut",
            (0.0025, depth * 3, (end - start) * full_len),
            (
                local_x + geo.FILLET_LEG / 2 + 0.001,
                local_y - 0.002,
                along((start + end) / 2),
            ),
            color=(0.10, 0.095, 0.09),
        )

    if plan.hole is not None:
        at, radius = plan.hole
        # Burn-through is a hole melted clean through, so the weld metal is absent
        # and what shows is the dark cavity behind it.
        ellipsoid(
            f"{path}/burn_through",
            (radius, radius, radius),
            (local_x - 0.002, local_y + 0.002, along(at)),
            color=(0.03, 0.03, 0.035),
        )

def build_lighting() -> None:
    """Even, diffuse light.

    A bright key against a dim dome throws hard shadows, and anything the key
    misses drops to black: on the first pass 84 percent of the weld bead itself
    rendered as pure black. An inspection station is diffusely lit precisely so
    that a bead reads everywhere along its length, so the dome carries the scene
    and the rectangles only lift it.
    """
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr().Set(3000.0)

    for name, position, intensity in (
        ("Key", (geo.FIXTURE_X, -2.0, 3.0), 3500.0),
        ("Back", (geo.FIXTURE_X, 2.0, 2.6), 2000.0),
        ("FarmFill", (BEAD_FARM_X, -1.2, geo.TABLE_TOP + 1.6), 3000.0),
        ("FarmSide", (BEAD_FARM_X - 1.2, 0.2, geo.TABLE_TOP + 1.0), 2000.0),
    ):
        light = UsdLux.RectLight.Define(stage, f"/World/{name}")
        light.CreateIntensityAttr().Set(intensity)
        light.CreateWidthAttr().Set(3.0)
        light.CreateHeightAttr().Set(3.0)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*position))


def describe_camera(path: str, focal_length: float = 24.0) -> dict:
    """Everything needed to turn this camera's depth back into world points.

    USD composes with row vectors, so its local-to-world is the transpose of the
    column-vector matrix numpy expects. Getting that backwards silently mirrors the
    whole point cloud.
    """
    transform = UsdGeom.Xformable(
        stage.GetPrimAtPath(path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return {
        "width": RESOLUTION[0],
        "height": RESOLUTION[1],
        "focal_length_mm": focal_length,
        "horizontal_aperture_mm": APERTURE_MM,
        "vertical_aperture_mm": APERTURE_MM,
        "to_world": [[float(transform[j][i]) for j in range(4)] for i in range(4)],
    }


def make_camera(path: str, eye, aim, focal_length: float = 24.0):
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(focal_length)
    # Square the aperture to match the square render product. Leaving the default
    # 4:3 aperture against a 1:1 target is how a subject that the geometry says is
    # centred ends up outside the frame.
    camera.CreateHorizontalApertureAttr().Set(APERTURE_MM)
    camera.CreateVerticalApertureAttr().Set(APERTURE_MM)
    xform = UsdGeom.Xformable(camera)
    op = xform.AddTransformOp()
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*aim), Gf.Vec3d(0, 0, 1))
    op.Set(view.GetInverse())
    return camera, op


# --------------------------------------------------------------------- arrivals


def settle_part(part: SingleRigidPrim, spawn_y: float, spawn_yaw_deg: float) -> tuple:
    """Drive one part down the line until it rests on the stop. Returns its pose.

    Short repeated pushes rather than one shove: the contact solver needs steps
    between nudges, and a single hard kick either stalls short or rebounds.
    """
    yaw = math.radians(spawn_yaw_deg)
    quaternion = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    part.set_world_pose(
        position=np.array([1.2, spawn_y, geo.TABLE_TOP + 0.002]),
        orientation=quaternion,
    )
    part.set_linear_velocity(np.zeros(3))
    part.set_angular_velocity(np.zeros(3))

    last_x = 0.0
    for _ in range(90):
        position, _ = part.get_world_pose()
        if position[0] > geo.FIXTURE_X - 0.02:
            break
        if abs(position[0] - last_x) < 1e-4 and position[0] > 3.0:
            break
        last_x = float(position[0])
        part.set_linear_velocity(np.array([1.2, 0.0, 0.0]))
        part.set_angular_velocity(np.zeros(3))
        for _ in range(12):
            world.step(render=False)

    for _ in range(120):  # let it come to rest against the stop
        world.step(render=False)

    position, orientation = part.get_world_pose()
    w, _, _, z = orientation
    settled_yaw = 2.0 * math.atan2(float(z), float(w))
    return float(position[0]), float(position[1]), settled_yaw


# ------------------------------------------------------------------------- main


def report_bounds(paths: list[str]) -> None:
    """Print measured world extents. Authored numbers and rendered geometry disagree
    often enough that every layout claim should be checked against a bbox."""
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    print(f"{'prim':34} {'x range':>20} {'y range':>20} {'z range':>20}")
    print("-" * 98)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            print(f"{path:34} MISSING")
            continue
        box_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box_range.IsEmpty():
            print(f"{path:34} EMPTY BOUND")
            continue
        purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
        lo, hi = box_range.GetMin(), box_range.GetMax()
        print(f"{'':34} purpose={purpose}")
        print(
            f"{path:34} "
            f"{lo[0]:8.3f}..{hi[0]:<8.3f} "
            f"{lo[1]:8.3f}..{hi[1]:<8.3f} "
            f"{lo[2]:8.3f}..{hi[2]:<8.3f}"
        )


def main() -> int:
    rng = random.Random(ARGS.seed)
    plans = defect_lib.plan_campaign(ARGS.trials, seed=ARGS.seed, defect_rate=ARGS.defect_rate)

    build_fixture()
    build_lighting()
    build_part("/World/Part", physics=True)
    build_bead("/World/Part/Bead", defect_lib.clean_bead())

    world.reset()
    part = SingleRigidPrim("/World/Part")
    part.initialize()

    if ARGS.bounds:
        print("\nExpected: part half-width %.3f, stop inner face %.3f, so a seated "
              "part centres at %.3f\n" % (
                  geo.PART_W / 2,
                  geo.FIXTURE_X + 0.485 - 0.025,
                  geo.FIXTURE_X + 0.485 - 0.025 - geo.PART_W / 2,
              ))
        report_bounds([
            "/World/Fix/Table",
            "/World/Fix/Stop",
            "/World/Fix/Slide",
            "/World/Guide_0",
            "/World/Guide_1",
            "/World/Guide_0",
            "/World/Part",
        ])
        return 0

    # --- phase 1: arrivals under physics, no rendering ----------------------
    print(f"[station] settling {ARGS.trials} arrivals", flush=True)
    trials = []
    for index in range(ARGS.trials):
        spawn_y = rng.uniform(-0.030, 0.030)
        spawn_yaw = rng.uniform(-2.0, 2.0)
        x, y, yaw = settle_part(part, spawn_y, spawn_yaw)
        joint = geo.true_joint(x, y, yaw)
        trials.append(
            {
                "trial": index,
                "spawn_dy_mm": spawn_y * 1000.0,
                "spawn_yaw_deg": spawn_yaw,
                "settled": [x, y, math.degrees(yaw)],
                "true_joint": list(joint),
                "blind_error_mm": geo.blind_error_mm(x, y, yaw),
                "injected_defect": plans[index].defect,
                "defect_severity": plans[index].severity,
            }
        )
        print(
            f"  trial {index:3d}  settled x={x:.4f} y={y:+.4f} yaw={math.degrees(yaw):+.2f}deg"
            f"  blind miss {trials[-1]['blind_error_mm']:.1f} mm"
            f"  bead: {plans[index].describe()}",
            flush=True,
        )

    if ARGS.no_render:
        write_manifest(trials, rendered=False)
        print("[station] physics only, no images written")
        return 0

    # --- phase 2: pre-weld scans through the replicator graph ---------------
    # The part is parked and a render-only twin is driven instead, so the pose
    # on camera is exactly the settled pose rather than whatever physics is
    # doing at capture time.
    part.set_world_pose(position=np.array(PART_PARK), orientation=np.array([1.0, 0, 0, 0]))
    build_part("/World/PartVis", physics=False)
    build_bead("/World/PartVis/Bead", defect_lib.clean_bead(), tag_semantics=False)

    # The bead farm has to exist before the first frame is drawn. Geometry created
    # after the stage has composed never reaches the headless renderer, so beads
    # built later photograph as an empty table. Each part's bead is distinct
    # geometry and cannot be swapped, so all of them are built now at their own
    # stations and the camera visits each in turn.
    print("[station] building bead farm", flush=True)
    farm_root = UsdGeom.Xform.Define(stage, "/World/BeadFarm")
    farm_root.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    farm_root.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))
    for index, plan in enumerate(plans):
        holder = UsdGeom.Xform.Define(stage, f"/World/BeadFarm/part_{index}")
        holder.AddTranslateOp().Set(
            Gf.Vec3d(BEAD_FARM_X, index * BEAD_FARM_PITCH, geo.TABLE_TOP)
        )
        build_bead(
            f"/World/BeadFarm/part_{index}/bead",
            plan,
            detailed=True,
            seed=ARGS.seed * 1000 + index,
        )
        # Two panels meeting at the joint, in the same orientation as the corner on
        # the real part. Without them the bead photographs as a stick floating in
        # space, and no inspector, human or otherwise, could call it a weld.
        local_x, local_y = geo.JOINT_LOCAL
        run = 0.30
        box(
            f"/World/BeadFarm/part_{index}/panel_side",
            (geo.PART_T, run, geo.PART_D),
            (local_x - geo.PART_T / 2, local_y + run / 2, geo.PART_D / 2),
            color=(0.58, 0.60, 0.64),
        )
        box(
            f"/World/BeadFarm/part_{index}/panel_front",
            (run, geo.PART_T, geo.PART_D),
            (local_x - run / 2, local_y + geo.PART_T / 2, geo.PART_D / 2),
            color=(0.58, 0.60, 0.64),
        )

    # Both views are captured in the same frame. The part twin is driven to the
    # settled pose for the wide scan, and the bead farm slides so that this part's
    # bead sits at a fixed inspection spot for the close-up. Two cameras, both
    # composed before the first frame, photograph the two at once.
    #
    # The camera cannot be re-aimed between frames. Its transform updates in USD
    # and the renderer ignores it, exactly as scene geometry does. Only poses
    # driven through the replicator graph re-render, so the subject moves to the
    # camera rather than the other way round.
    # Candidate viewing angles. The camera cannot move, so the part turns about the
    # joint axis instead, which leaves the joint itself exactly where it is. The
    # target is therefore identical from every view and any difference between
    # them is perception, not the thing being measured.
    if ARGS.views == 1:
        orbit_angles = [0.0]
    else:
        step = 2.0 * ARGS.orbit_range / (ARGS.views - 1)
        orbit_angles = [round(-ARGS.orbit_range + i * step, 3) for i in range(ARGS.views)]
    print(f"[station] candidate views (degrees): {orbit_angles}", flush=True)

    part_positions, part_rotations = [], []
    farm_positions = []
    for index, row in enumerate(trials):
        x, y, yaw_deg = row["settled"]
        joint_x, joint_y = row["true_joint"]
        row["views"] = []
        for angle in orbit_angles:
            px, py, pyaw = geo.orbit_about_joint(
                x, y, math.radians(yaw_deg), joint_x, joint_y, math.radians(angle)
            )
            # Two frames per capture: the writer trails the scene by one.
            part_positions += [(px, py, geo.TABLE_TOP + 0.003)] * 2
            part_rotations += [(0.0, 0.0, math.degrees(pyaw))] * 2
            # The bead does not change with the view, so the farm holds station.
            farm_positions += [(0.0, -index * BEAD_FARM_PITCH, 0.0)] * 2
            row["views"].append({"orbit_deg": angle})

    if ARGS.xray:
        painted = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Gprim):
                name = str(prim.GetPath()).replace("/World/", "").replace("/", ".")
                UsdSemantics.LabelsAPI.Apply(prim, "class").GetLabelsAttr().Set([name])
                painted += 1
        print(f"[station] x-ray: labelled {painted} prims", flush=True)

    twin_group = rep.get.prims(path_pattern="^/World/PartVis$")
    farm_group = rep.get.prims(path_pattern="^/World/BeadFarm$")
    with rep.trigger.on_frame():
        with twin_group:
            rep.modify.pose(
                position=rep.distribution.sequence(part_positions),
                rotation=rep.distribution.sequence(part_rotations),
            )
        with farm_group:
            rep.modify.pose(position=rep.distribution.sequence(farm_positions))

    scan_camera, _ = make_camera("/World/ScanCam", geo.CAMERA_EYE, geo.CAMERA_AIM)
    # The scan framing was found empirically and is known to work. The bead sits at
    # the same offset inside the farm holder that the joint sits at inside the part,
    # so translating that proven view onto the farm reproduces it exactly. A longer
    # lens then tightens onto the bead without moving the camera off a known-good
    # line of sight.
    shift = (BEAD_FARM_X - geo.FIXTURE_X, 0.0, 0.0)
    bead_eye = tuple(v + s for v, s in zip(geo.CAMERA_EYE, shift, strict=True))
    bead_target = tuple(v + s for v, s in zip(geo.CAMERA_AIM, shift, strict=True))
    bead_camera, _ = make_camera("/World/BeadCam", bead_eye, bead_target, focal_length=35.0)

    scan_product = rep.create.render_product(str(scan_camera.GetPath()), RESOLUTION)
    bead_product = rep.create.render_product(str(bead_camera.GetPath()), RESOLUTION)

    scan_writer = rep.WriterRegistry.get("BasicWriter")
    scan_writer.initialize(
        output_dir=str(OUT_DIR / "scans").replace("\\", "/"),
        rgb=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=False,
        # Depth and normals come free from the same render, and they are the
        # control inputs a world model needs to turn these renders into
        # photorealistic variants. Capturing them now costs nothing and saves
        # re-running the whole campaign later.
        distance_to_image_plane=True,
        normals=True,
    )
    scan_writer.attach([scan_product])

    bead_writer = rep.WriterRegistry.get("BasicWriter")
    bead_writer.initialize(
        output_dir=str(OUT_DIR / "beads").replace("\\", "/"),
        rgb=True,
        semantic_segmentation=ARGS.xray,
        colorize_semantic_segmentation=False,
    )
    bead_writer.attach([bead_product])

    total = len(trials) * len(orbit_angles)
    print(f"[station] capturing {total} views over {len(trials)} parts", flush=True)
    frame = 0
    for row in trials:
        for view in row["views"]:
            rep.orchestrator.step(rt_subframes=ARGS.subframes)  # lag guard, discarded
            rep.orchestrator.step(rt_subframes=ARGS.subframes)  # matches the pose
            frame += 2
            stamp = frame - 1
            view["rgb"] = "scans/rgb_%04d.png" % stamp
            view["mask"] = "scans/semantic_segmentation_%04d.png" % stamp
            view["depth"] = "scans/distance_to_image_plane_%04d.npy" % stamp
            view["normals"] = "scans/normals_%04d.png" % stamp
        # The bead is the same whichever way the part was turned, so one shot of it.
        row["bead_image"] = "beads/rgb_%04d.png" % (frame - 1)

    rep.orchestrator.wait_until_complete()
    camera = describe_camera("/World/ScanCam")
    eye = [round(camera["to_world"][i][3], 4) for i in range(3)]
    print(f"[station] scan camera at {eye}, authored {geo.CAMERA_EYE}", flush=True)
    write_manifest(trials, rendered=True, camera=camera)
    return 0


def write_manifest(trials: list[dict], rendered: bool, camera: dict | None = None) -> None:
    client = None
    if not ARGS.no_post:
        client = CellClient(base_url=ARGS.api)
        if not client.healthy():
            print("[station] service not reachable, parts will be queued to disk", flush=True)

    for row in trials:
        if client is not None:
            row["part_serial"] = client.create_part(
                station="sim",
                spawn_dy_mm=row["spawn_dy_mm"],
                spawn_yaw_deg=row["spawn_yaw_deg"],
            )

    manifest = {
        "trials": ARGS.trials,
        "seed": ARGS.seed,
        "defect_rate": ARGS.defect_rate,
        "rendered": rendered,
        "camera": camera,
        "joint_height": geo.TABLE_TOP + geo.PART_D / 2,
        "capture_dir": str(OUT_DIR),
        "rows": trials,
    }
    path = OUT_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[station] wrote {path}", flush=True)


if __name__ == "__main__":
    try:
        code = main()
    finally:
        app.close()
    raise SystemExit(code)
