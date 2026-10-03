"""Drive the crawler along the survey run and capture what it sees, with the answers.

Run inside Blender:

    python scripts/blender_send.py tank/sim/capture_tank.py

This is the step that turns a scene into a dataset. The crawler is walked along
the seam it was sent to inspect, and at every station three things are written
down: the picture its camera took, the depth its camera measured, and what is
actually there. The last one is the point. A frame with no answer beside it can
be looked at but not scored, and a demonstration that cannot be scored is a
demonstration.

What is recorded at each station, and why.

  the picture     what the inspection model is given, and nothing else. The
                  ground truth is never in the frame.
  the depth       what the seam finder is given. Metres from the camera, one
                  float per pixel, the same shape the welding cell already
                  reads.
  faults in view  each fault in the scene is projected through the camera, so
                  a verdict can be marked against what was actually in front of
                  the lens rather than against what was somewhere in the tank.
  the probe spot  where the ultrasonic probe is touching, and the true wall
                  thickness there, worked out from the thinning patches rather
                  than sampled off a grid. This is the answer the probe planner
                  is scored against, and on most stations it is the nominal
                  thickness, because most of a wall is fine.

Output goes to tank/data/survey/: frames, depth, and manifest.json.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import blender_tank  # noqa: E402

importlib.reload(blender_tank)

ROOT = HERE.parent.parent
OUT = ROOT / "tank" / "data" / "survey"
COMPOSITOR = "CaptureCompositor"
STATIONS = 24            # 4 m of seam at 174 mm a step, so frames overlap
RES = (1280, 720)
SAMPLES = 48

# Where the probe could be put at each station. The arm swings across the seam,
# so a station is not one place to measure but a row of them, and choosing
# between them is the decision the planner is scored on. Probing every one of
# them is the brute force answer and costs the whole point: a thirty metre wall
# cannot be probed everywhere, which is why the choice matters.
PROBE_OFFSETS_M = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)


def thickness_mm(defects, theta, z) -> tuple[float, list[str]]:
    """True wall thickness at a point, and which patches took metal off it.

    The thinning patches are parameters, not a grid, so this can be asked at
    any point on the wall rather than at the points somebody happened to
    sample. Distance is measured along the wall, not through the tank.
    """
    course = min(blender_tank.COURSES - 1, max(0, int(z // blender_tank.COURSE_H)))
    left = blender_tank.COURSE_MM[course]
    took = []
    for d in defects:
        if d["kind"] != "thinning":
            continue
        d_theta = ((theta - d["theta_rad"] + math.pi) % math.tau) - math.pi
        along = d_theta * blender_tank.RADIUS
        dist = math.hypot(along, z - d["z_m"])
        if dist < d["patch_radius_m"]:
            left -= d["loss_mm"] * math.cos(math.pi / 2 * dist / d["patch_radius_m"])
            took.append(d["id"])
    return round(left, 3), took


def setup_compositor():
    """A compositor whose output can be switched between picture and depth.

    Three things changed under this in Blender 5. The compositor is a node
    group the scene points at rather than a tree on the scene. There is no
    Composite node any more: the group output is what the render becomes. And
    the file output node will only write multilayer EXR, which Blender itself
    will not read back through the image API, so it is not used at all.

    So each station is rendered twice, once with the picture on the output and
    once with the depth. It costs a second a frame and it is the only route
    that produces a depth array this project can read without a new dependency.
    """
    scene = bpy.context.scene
    bpy.context.view_layer.use_pass_z = True
    group = (bpy.data.node_groups.get(COMPOSITOR)
             or bpy.data.node_groups.new(COMPOSITOR, "CompositorNodeTree"))
    group.nodes.clear()
    scene.use_nodes = True
    scene.compositing_node_group = group
    layers = group.nodes.new("CompositorNodeRLayers")
    sink = group.nodes.new("NodeGroupOutput")
    sink.location = (400, 0)
    return group, layers, sink


def route(group, layers, sink, which: str):
    """Put either the picture or the depth on the group output."""
    for link in list(group.links):
        group.links.remove(link)
    socket = layers.outputs.get(which) or layers.outputs.get("Z")
    if socket is None:
        raise RuntimeError(f"no {which} pass in this build")
    group.links.new(socket, sink.inputs[0])


def render_to(path: Path, fmt: str, depth_bits: str = "8"):
    scene = bpy.context.scene
    scene.render.filepath = str(path.with_suffix(""))
    scene.render.image_settings.file_format = fmt
    if fmt.startswith("OPEN_EXR"):
        scene.render.image_settings.color_depth = depth_bits
        scene.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(write_still=True)


def depth_to_npy(exr: Path, npy: Path) -> dict:
    """Read the rendered depth back and store it the way the cell already reads it.

    Downstream this project has always taken depth as a plain float array, so
    the conversion happens once here rather than every consumer growing an EXR
    reader.
    """
    image = bpy.data.images.load(str(exr))
    w, h = image.size
    channels = image.channels
    if channels == 0:
        bpy.data.images.remove(image)
        raise RuntimeError(f"{exr.name} came back with no channels")
    buf = np.empty(w * h * channels, dtype=np.float32)
    image.pixels.foreach_get(buf)
    depth = buf.reshape(h, w, channels)[:, :, 0][::-1].copy()
    bpy.data.images.remove(image)
    exr.unlink(missing_ok=True)
    # Anything the camera did not hit comes back at the far clip.
    sky = depth > 100.0
    depth[sky] = np.nan
    np.save(npy, depth)
    return {"shape": [int(h), int(w)],
            "min_m": None if np.all(sky) else round(float(np.nanmin(depth)), 4),
            "max_m": None if np.all(sky) else round(float(np.nanmax(depth)), 4),
            "sky_fraction": round(float(sky.mean()), 4)}


def footprint(d) -> list[tuple[float, float]]:
    """Points spread over the whole of a fault, in wall coordinates.

    A corrosion patch is three quarters of a metre across and a frame is a
    third of one, so a patch whose centre is a metre away still fills the
    corner of the picture. Projecting the centre alone said it was not there,
    and the inspector was marked wrong twice for naming rust it could plainly
    see. The truth has to cover the thing, not point at the middle of it.
    """
    theta, z = d["theta_rad"], d["z_m"]
    if "arc_from" in d:                       # runs along a circumferential seam
        lo, hi = d["arc_from"], d["arc_to"]
        return [(lo + (hi - lo) * i / 8, z) for i in range(9)]
    if d["kind"] == "thinning":               # a patch, and islands beyond it
        r = d["patch_radius_m"]
        points = [(theta, z)]
        for k in range(12):
            a = math.tau * k / 12
            for reach in (0.6, 1.0, 1.7):
                points.append((theta + math.cos(a) * r * reach / blender_tank.RADIUS,
                               z + math.sin(a) * r * reach))
        # The islands themselves, where the scene actually put them, rather
        # than a ring at the radius they roughly live on.
        points.extend((it, iz) for it, iz, _ in (d.get("rust_islands") or []))
        return points
    if "length_mm" in d:                      # runs up a vertical seam
        half = d["length_mm"] / 2000
        return [(theta, z - half + 2 * half * i / 6) for i in range(7)]
    if d["kind"] == "porosity":               # a short cluster of pores
        return [(theta + (i - 2) * 0.014 / blender_tank.RADIUS, z) for i in range(5)]
    return [(theta, z)]


# How wide each kind of fault actually is, half width in metres, across the
# line it runs along. One shared figure made a crack as wide as the bead.
HALF_WIDTH_M = {"crack": 0.002, "lack_of_fusion": 0.006, "undercut": 0.006,
                "burn_through": 0.009, "porosity": 0.012}


def covers(d):
    """A test for whether a point on the wall is inside this fault."""
    theta, z, R = d["theta_rad"], d["z_m"], blender_tank.RADIUS
    hw = HALF_WIDTH_M.get(d["kind"], 0.010)
    if "arc_from" in d:                       # runs along a circumferential seam
        lo, hi = d["arc_from"], d["arc_to"]
        return lambda t, y: lo <= t <= hi and abs(y - z) <= hw
    if d["kind"] == "thinning":               # a patch, and its islands
        r = d["patch_radius_m"] * 0.70        # the mean reach of the ragged edge
        islands = d.get("rust_islands") or []

        def rusted(t, y):
            if math.hypot((t - theta) * R, y - z) <= r:
                return True
            return any(math.hypot((t - it) * R, y - iz) <= ir
                       for it, iz, ir in islands)

        return rusted
    if "length_mm" in d:                      # runs up a vertical seam
        half = d["length_mm"] / 2000
        return lambda t, y: abs(y - z) <= half and abs((t - theta) * R) <= hw
    if d["kind"] == "porosity":               # a short cluster of pores
        return lambda t, y: abs((t - theta) * R) <= 0.035 and abs(y - z) <= hw
    return lambda t, y: math.hypot((t - theta) * R, y - z) <= 0.02


def wall_in_frame(scene, cam, centre_theta, centre_z) -> list[tuple[float, float]]:
    """A grid of wall points the lens can actually see, sampled once a station.

    Computed here rather than per fault because it does not depend on the
    fault, and every fault at the station is then a cheap test against it.
    """
    R = blender_tank.RADIUS
    shown = []
    # Twenty millimetre steps. At thirty the grid walked over rust islands
    # small enough to fit between two samples and reported clean plate.
    for i in range(37):
        t = centre_theta + (i - 18) * 0.020 / R
        for j in range(25):
            y = centre_z + (j - 12) * 0.016
            world = blender_tank.wall_point(t, y, blender_tank.BEAD_R)
            uvw = world_to_camera_view(scene, cam, Vector(world))
            if uvw.z > 0 and 0.0 <= uvw.x <= 1.0 and 0.0 <= uvw.y <= 1.0:
                shown.append((t, y))
    return shown


def faults_in_view(scene, cam, defects, centre_theta, centre_z) -> list[dict]:
    """Every fault the lens is pointing at, and how much of the picture it is."""
    shown = wall_in_frame(scene, cam, centre_theta, centre_z)
    seen = []
    for d in defects:
        points = footprint(d)
        inside = []
        for theta, z in points:
            world = blender_tank.wall_point(theta, z, blender_tank.BEAD_R)
            uvw = world_to_camera_view(scene, cam, Vector(world))
            if uvw.z > 0 and 0.0 <= uvw.x <= 1.0 and 0.0 <= uvw.y <= 1.0:
                inside.append(uvw)
        here = covers(d)
        share = (sum(1 for t, y in shown if here(t, y)) / len(shown)) if shown else 0.0
        if not inside and share <= 0.0:
            continue
        if inside:
            middle = inside[len(inside) // 2]
            u, v, rng_m = float(middle.x), float(1.0 - middle.y), float(middle.z)
        else:
            u, v, rng_m = 0.5, 0.5, 0.0
        seen.append({"id": d["id"], "kind": d["kind"],
                     "u": round(u, 4), "v": round(v, 4),
                     "range_m": round(rng_m, 4),
                     "of_the_fault": round(len(inside) / len(points), 3),
                     "of_the_frame": round(share, 3),
                     "coverage": round(max(len(inside) / len(points), share), 3),
                     "visible_to_a_camera": d["visible"],
                     "rejectable": bool(d.get("rejectable"))})
    return seen


def main() -> dict:
    built = blender_tank.main()
    truth = json.loads(Path(built["ground_truth"]).read_text(encoding="utf-8"))
    defects = truth["defects"]
    run = truth["survey_run"]

    frames = OUT / "frames"
    depths = OUT / "depth"
    for folder in (frames, depths):
        folder.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = RES
    with contextlib.suppress(AttributeError, TypeError):
        scene.eevee.taa_render_samples = SAMPLES
    group, layers, sink = setup_compositor()

    rig = bpy.data.objects["Crawler"]
    cam = bpy.data.objects["Crawler_Cam"]
    scene.camera = cam

    lo, hi = run["theta_from_rad"], run["theta_to_rad"]
    z = run["z_m"]
    records = []
    for index in range(STATIONS):
        theta = lo + (hi - lo) * index / max(1, STATIONS - 1)
        rig.matrix_world = blender_tank.crawler_matrix(theta, z)
        bpy.context.view_layer.update()

        scene.frame_set(index + 1)
        rgb = frames / f"rgb_{index:04d}.png"
        route(group, layers, sink, "Image")
        render_to(rgb, "PNG")

        exr = depths / f"depth_{index:04d}.exr"
        npy = depths / f"depth_{index:04d}.npy"
        route(group, layers, sink, "Depth")
        render_to(exr, "OPEN_EXR", depth_bits="32")
        stats = depth_to_npy(exr, npy)

        # The probe rides under the deck at x = 0.10 in the crawler's frame and
        # swings across the seam, so each station offers a row of places to
        # measure. Every one is recorded with the answer, and the planner is
        # scored on which it picks.
        candidates = []
        for offset in PROBE_OFFSETS_M:
            contact = rig.matrix_world @ Vector((0.10, -offset, 0.0))
            c_theta = math.atan2(contact.y, contact.x)
            left, took = thickness_mm(defects, c_theta, contact.z)
            course = min(blender_tank.COURSES - 1,
                         max(0, int(contact.z // blender_tank.COURSE_H)))
            nominal = blender_tank.COURSE_MM[course]
            candidates.append({
                "offset_m": offset,
                "theta_rad": round(c_theta, 6), "z_m": round(contact.z, 4),
                "nominal_mm": nominal, "true_thickness_mm": left,
                "remaining_fraction": round(left / nominal, 4),
                "thinned_by": took,
                "rejectable": left / nominal < truth["acceptance"]["thinning"][
                    "limit_fraction"],
            })
        on_seam = next(c for c in candidates if c["offset_m"] == 0.0)

        records.append({
            "station": index,
            "theta_rad": round(theta, 6), "z_m": z,
            "rgb": f"frames/{rgb.name}", "depth": f"depth/{npy.name}",
            "depth_stats": stats,
            "faults_in_view": faults_in_view(scene, cam, defects, theta, z),
            "probe_candidates": candidates,
            "probe_on_seam": on_seam,
        })
        thin = [c for c in candidates if c["thinned_by"]]
        print(f"  station {index:>2}  faults in view "
              f"{len(records[-1]['faults_in_view'])}  thin probe points "
              f"{len(thin)}/{len(candidates)}", flush=True)

    manifest = {
        "what_this_is": "one pass of the crawler along the seam it was sent to",
        "stations": STATIONS,
        "step_m": round((hi - lo) * blender_tank.RADIUS / max(1, STATIONS - 1), 4),
        "resolution": list(RES),
        "camera": truth["crawler"],
        "probe_offsets_m": list(PROBE_OFFSETS_M),
        "survey_run": run,
        "acceptance": truth["acceptance"],
        "tank": truth["tank"],
        "defects": defects,
        "thickness_model": truth["thickness_model"],
        "frames": records,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    points = [c for r in records for c in r["probe_candidates"]]
    thin = [c for c in points if c["thinned_by"]]
    return {
        "stations": STATIONS,
        "step_mm": round(manifest["step_m"] * 1000, 1),
        "frames_with_a_fault_in_view": sum(1 for r in records if r["faults_in_view"]),
        "probe_points_offered": len(points),
        "probe_points_over_thinned_wall": len(thin),
        "probe_points_rejectable": sum(1 for c in points if c["rejectable"]),
        "thinnest_wall_mm": min(c["true_thickness_mm"] for c in points),
        "faults_on_the_run": run["faults_on_it"],
        "manifest": str(OUT / "manifest.json"),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
