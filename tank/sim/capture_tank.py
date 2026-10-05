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
ENGINE = "BLENDER_EEVEE"   # what the scored dataset was rendered with

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


# The answer key is computed by tank/truth.py, which needs no Blender, so the
# capture, the scorer, the dashboard and the tests all apply one rule. It used
# to be computed here, projecting every fault as if it sat on the crown of the
# bead, and at station 12 that put a crack outside a frame it was plainly in.
# What the capture owes the rule is what only the scene knows: the vertices of
# every fault mesh on the run, and proof that truth.py's camera still projects
# exactly the way Blender's does.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tank import truth  # noqa: E402

importlib.reload(truth)


def export_fault_geometry(run) -> int:
    """Write the drawn vertices of the faults on the run, in wall coordinates."""
    radius = blender_tank.RADIUS
    on_run = set(run["faults_on_it"])
    faults, meshes = {}, {}
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        if obj.type != "MESH" or not obj.name.startswith("Defect_"):
            continue
        fid = obj.name.split("_")[1]
        if fid not in on_run:
            continue
        meshes.setdefault(fid, []).append(obj.name)
        points = faults.setdefault(fid, [])
        for v in obj.data.vertices:
            w = obj.matrix_world @ v.co
            points.append([round(math.atan2(w.y, w.x), 7), round(w.z, 6),
                           round(radius - math.hypot(w.x, w.y), 6)])
    # The edge of every rust patch a camera can see, and of each island: the
    # outermost ring of each mesh, in the order it was built.
    outlines = {}
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        if obj.type != "MESH" or not (obj.name.startswith("Defect_")
                                      and obj.name.endswith("_rust")):
            continue
        fid = obj.name.split("_")[1]
        around, out = blender_tank.RUST_AROUND, blender_tank.RUST_OUT

        def ring_point(o, index):
            w = o.matrix_world @ o.data.vertices[index].co
            return [round(math.atan2(w.y, w.x), 7), round(w.z, 6),
                    round(radius - math.hypot(w.x, w.y), 6)]

        rings = [[ring_point(obj, i * (out + 1) + out) for i in range(around + 1)]]
        meshes[fid] = [obj.name]
        islands = bpy.data.objects.get(f"Defect_{fid}_freckles")
        if islands is not None:
            steps, deep = blender_tank.ISLAND_STEPS, blender_tank.ISLAND_RINGS
            per = (steps + 1) * (deep + 1)
            for k in range(len(islands.data.vertices) // per):
                rings.append([ring_point(islands, k * per + i * (deep + 1) + deep)
                              for i in range(steps + 1)])
            meshes[fid].append(islands.name)
        outlines[fid] = rings
    with truth.GEOMETRY.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"what_this_is": truth.GEOMETRY_NOTE, "meshes": meshes,
                                 "faults": faults, "outlines": outlines},
                                separators=(",", ":")))
    truth.reload_geometry()
    return sum(len(v) for v in faults.values()) + sum(
        len(ring) for rings in outlines.values() for ring in rings)


def check_projection(scene, cam, theta, z) -> float:
    """Stop the capture if truth.py and Blender disagree about the camera."""
    worst = 0.0
    for i in range(-3, 4):
        for inset in (-0.0016, 0.0, 0.0024, 0.008):
            t = theta + 0.47 / blender_tank.RADIUS + i * 0.06 / blender_tank.RADIUS
            world = blender_tank.wall_point(t, z + 0.04, inset)
            want = world_to_camera_view(scene, cam, Vector(world))
            x, y, _ = truth.project(world, theta, z)
            worst = max(worst, abs(x - want.x) * RES[0], abs(y - want.y) * RES[1])
    if worst > 0.05:
        raise RuntimeError(f"tank/truth.py projects {worst:.3f} px away from "
                           f"Blender at theta {theta:.5f}; the key would be wrong")
    return worst


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
    # Pin the renderer. The dataset is the evidence, and evidence that changes
    # appearance because somebody improved the lighting an hour ago is not
    # evidence. tank/sim/look.py switches this scene to Cycles and rebuilds
    # every material for the video, and without this line a capture run after
    # it would quietly produce a different dataset from the one every number
    # in the README was measured on, with nothing to say it had happened.
    #
    # Re-capturing under the Cycles look is the obvious next experiment and it
    # would be a harder test, because those frames are closer to a photograph.
    # It also invalidates every score until they are re-run, so it is a
    # decision rather than a side effect.
    scene.render.engine = ENGINE
    scene.render.resolution_x, scene.render.resolution_y = RES
    with contextlib.suppress(AttributeError, TypeError):
        scene.eevee.taa_render_samples = SAMPLES
    group, layers, sink = setup_compositor()

    rig = bpy.data.objects["Crawler"]
    cam = bpy.data.objects["Crawler_Cam"]
    scene.camera = cam

    lo, hi = run["theta_from_rad"], run["theta_to_rad"]
    z = run["z_m"]
    drawn = export_fault_geometry(run)
    print(f"  {drawn} drawn fault vertices written to {truth.GEOMETRY.name}", flush=True)
    records = []
    for index in range(STATIONS):
        theta = lo + (hi - lo) * index / max(1, STATIONS - 1)
        rig.matrix_world = blender_tank.crawler_matrix(theta, z)
        bpy.context.view_layer.update()
        check_projection(scene, cam, theta, z)

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
            "faults_in_view": truth.faults_in_view(defects, round(theta, 6), z),
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
