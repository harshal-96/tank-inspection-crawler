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
BEAD_FARM_PITCH = 1.0
PART_PARK = (-20.0, 0.0, geo.PART_D / 2 + 0.005)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()


# --------------------------------------------------------------------------- scene


def box(path: str, size, translate, color=(0.55, 0.57, 0.6)) -> "UsdGeom.Cube":
    prim = UsdGeom.Cube.Define(stage, path)
    prim.GetSizeAttr().Set(1.0)
    prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
    prim.AddScaleOp().Set(Gf.Vec3f(*size))
    prim.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
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
    slide_top = geo.TABLE_TOP + 0.0005
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
    # Ground.
    static_box("/World/Ground", (40.0, 40.0, 0.02), (0.0, 0.0, -0.01), color=(0.22, 0.22, 0.24))


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
    """The sheet-metal part: four walls, so the joint corner is a real corner."""
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(*PART_PARK))
    root.AddRotateXYZOp().Set(Gf.Vec3f(0, 0, 0))

    w, h, d, t = geo.PART_W, geo.PART_H, geo.PART_D, geo.PART_T
    panels = {
        "front": ((w, t, d), (0, -(h / 2 - t / 2), 0)),
        "back": ((w, t, d), (0, (h / 2 - t / 2), 0)),
        "left": ((t, h - 2 * t, d), (-(w / 2 - t / 2), 0, 0)),
        "right": ((t, h - 2 * t, d), ((w / 2 - t / 2), 0, 0)),
        "bottom": ((w, h, t), (0, 0, -(d / 2 - t / 2))),
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


def build_bead(path: str, plan, tag_semantics: bool = False) -> None:
    """Weld bead for one part, built from its plan so the defect is real geometry.

    The bead stands proud of both panel faces. A bead flush with the corner is
    enclosed by the panels and photographs as nothing at all.
    """
    UsdGeom.Xform.Define(stage, path)
    local_x, local_y = geo.JOINT_LOCAL
    bw, bd, full_len = geo.BEAD_SIZE

    for index, (start, end) in enumerate(plan.segments):
        length = (end - start) * full_len
        if length <= 1e-4:
            continue
        centre_z = geo.BEAD_Z - full_len / 2 + (start + end) / 2 * full_len
        segment = box(
            f"{path}/bead_{index}",
            (bw, bd, length),
            (local_x, local_y, centre_z),
            color=(0.78, 0.68, 0.48),
        )
        if tag_semantics:
            UsdSemantics.LabelsAPI.Apply(segment.GetPrim(), "class").GetLabelsAttr().Set(
                ["joint_target"]
            )

    for index, blob in enumerate(plan.blobs):
        sphere = UsdGeom.Sphere.Define(stage, f"{path}/blob_{index}")
        sphere.GetRadiusAttr().Set(blob.radius)
        z = geo.BEAD_Z - full_len / 2 + blob.along * full_len
        # Pores sit in the bead face; spatter lands outboard on the parent metal.
        sphere.AddTranslateOp().Set(Gf.Vec3d(local_x + blob.offset, local_y + blob.offset, z))
        dark = (0.10, 0.09, 0.08) if blob.kind == defect_lib.POROSITY else (0.72, 0.66, 0.52)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*dark)])

    if plan.groove is not None:
        start, end, depth = plan.groove
        length = (end - start) * full_len
        centre_z = geo.BEAD_Z - full_len / 2 + (start + end) / 2 * full_len
        groove = box(
            f"{path}/undercut",
            (0.006, depth * 4, length),
            (local_x - bw / 2 - 0.003, local_y, centre_z),
            color=(0.12, 0.11, 0.10),
        )
        UsdGeom.Imageable(groove).CreatePurposeAttr().Set("default")

    if plan.hole is not None:
        along, radius = plan.hole
        hole = UsdGeom.Sphere.Define(stage, f"{path}/burn_through")
        hole.GetRadiusAttr().Set(radius)
        hole.AddTranslateOp().Set(
            Gf.Vec3d(local_x, local_y, geo.BEAD_Z - full_len / 2 + along * full_len)
        )
        hole.GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.05, 0.05)])


def build_lighting() -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr().Set(650.0)
    key = UsdLux.RectLight.Define(stage, "/World/Key")
    key.CreateIntensityAttr().Set(9000.0)
    key.CreateWidthAttr().Set(2.0)
    key.CreateHeightAttr().Set(2.0)
    UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(geo.FIXTURE_X, -2.0, 3.0))


def make_camera(path: str, eye, aim):
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr().Set(24.0)
    xform = UsdGeom.Xformable(camera)
    op = xform.AddTransformOp()
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*aim), Gf.Vec3d(0, 0, 1))
    op.Set(view.GetInverse())
    return camera, op


def aim_camera(op, eye, aim) -> None:
    """Re-aim by writing the transform value. This does reach the renderer."""
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*aim), Gf.Vec3d(0, 0, 1))
    op.Set(view.GetInverse())


# --------------------------------------------------------------------- arrivals


def settle_part(part: SingleRigidPrim, spawn_y: float, spawn_yaw_deg: float) -> tuple:
    """Drive one part down the line until it rests on the stop. Returns its pose.

    Short repeated pushes rather than one shove: the contact solver needs steps
    between nudges, and a single hard kick either stalls short or rebounds.
    """
    yaw = math.radians(spawn_yaw_deg)
    quaternion = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    part.set_world_pose(
        position=np.array([1.2, spawn_y, geo.TABLE_TOP + geo.PART_D / 2 + 0.002]),
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
        lo, hi = box_range.GetMin(), box_range.GetMax()
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
            "/World/Part",
            "/World/Part/front",
            "/World/Part/left",
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
    build_bead("/World/PartVis/Bead", defect_lib.clean_bead(), tag_semantics=True)

    positions, rotations = [], []
    for row in trials:
        x, y, yaw_deg = row["settled"]
        # Two frames per capture: the writer trails the scene by one.
        positions += [(x, y, geo.TABLE_TOP + 0.003)] * 2
        rotations += [(0.0, 0.0, yaw_deg)] * 2
    # Park the twin for the bead phase.
    positions += [PART_PARK] * (2 * ARGS.trials + 4)
    rotations += [(0.0, 0.0, 0.0)] * (2 * ARGS.trials + 4)

    twin_group = rep.get.prims(path_pattern="^/World/PartVis$")
    with rep.trigger.on_frame():
        with twin_group:
            rep.modify.pose(
                position=rep.distribution.sequence(positions),
                rotation=rep.distribution.sequence(rotations),
            )

    scan_camera, scan_op = make_camera("/World/ScanCam", geo.CAMERA_EYE, geo.CAMERA_AIM)
    product = rep.create.render_product(str(scan_camera.GetPath()), (640, 640))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(OUT_DIR).replace("\\", "/"),
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
    writer.attach([product])

    print("[station] capturing pre-weld scans", flush=True)
    frame = 0
    for row in trials:
        rep.orchestrator.step(rt_subframes=12)  # lag guard, discarded
        rep.orchestrator.step(rt_subframes=12)  # the one that matches the pose
        frame += 2
        row["scan_image"] = "rgb_%04d.png" % (frame - 1)
        row["scan_mask"] = "semantic_segmentation_%04d.png" % (frame - 1)
        row["scan_depth"] = "distance_to_image_plane_%04d.npy" % (frame - 1)
        row["scan_normals"] = "normals_%04d.png" % (frame - 1)

    # --- phase 3: post-weld beads -------------------------------------------
    # Each bead is distinct geometry, and geometry cannot be swapped once the
    # stage has composed. So every bead is built up front at its own station and
    # the camera visits them in turn.
    print("[station] building bead farm", flush=True)
    for index, plan in enumerate(plans):
        station_y = index * BEAD_FARM_PITCH
        holder = UsdGeom.Xform.Define(stage, f"/World/BeadFarm/part_{index}")
        holder.AddTranslateOp().Set(Gf.Vec3d(BEAD_FARM_X, station_y, geo.TABLE_TOP))
        build_bead(f"/World/BeadFarm/part_{index}/bead", plan)
        # A short length of parent metal behind the bead, so the shot reads as a joint.
        box(
            f"/World/BeadFarm/part_{index}/backing",
            (0.18, 0.02, geo.PART_D),
            (geo.JOINT_LOCAL[0] + 0.09, geo.JOINT_LOCAL[1], geo.BEAD_Z),
            color=(0.60, 0.62, 0.66),
        )

    print("[station] capturing post-weld beads", flush=True)
    for index, row in enumerate(trials):
        station_y = index * BEAD_FARM_PITCH
        target = (
            BEAD_FARM_X + geo.JOINT_LOCAL[0],
            station_y + geo.JOINT_LOCAL[1],
            geo.TABLE_TOP + geo.BEAD_Z,
        )
        eye = (target[0] - 0.55, target[1] - 0.45, target[2] + 0.30)
        aim_camera(scan_op, eye, target)
        rep.orchestrator.step(rt_subframes=12)
        rep.orchestrator.step(rt_subframes=12)
        frame += 2
        row["bead_image"] = "rgb_%04d.png" % (frame - 1)

    rep.orchestrator.wait_until_complete()
    write_manifest(trials, rendered=True)
    return 0


def write_manifest(trials: list[dict], rendered: bool) -> None:
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
