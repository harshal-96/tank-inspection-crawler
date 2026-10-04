"""The stills the video needs, rendered in one pass.

    python scripts/blender_send.py tank/sim/blender_tank.py
    python scripts/blender_send.py tank/sim/crawler_detail.py
    python scripts/blender_send.py tank/sim/look.py
    python scripts/blender_send.py tank/sim/stills.py

Four frames, each doing a job the script asks for and none of them decorative.

  establishing  far enough back that the shell reads as a tank and not a wall.
                This is the only shot that answers "where are we", and without
                it a viewer spends the first ten seconds working out the scale
                instead of listening.
  crawler       three quarters, close, so the machine reads as built: the
                bolts, the lugs, the gland, the lifting eyes.
  corroded      the patch the camera can see, which measures inside limits.
  clean         plate that looks like new steel, over the patch that does not.

The last two are the argument, and they only work as a pair. Shown alone
either one is a picture of a wall.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tank.sim import blender_tank as T  # noqa: E402
from tank.sim.meshkit import look_at  # noqa: E402

OUT = ROOT / "tank" / "data" / "shots"
SAMPLES = 400
RES = (1920, 1080)

# theta, z of the two patches, from the scene's own ground truth rather than
# typed in a second time and allowed to drift.
RUN_Z = 4.0


def _cam():
    obj = bpy.data.objects.get("Still_Cam")
    if obj is None:
        obj = bpy.data.objects.new("Still_Cam", bpy.data.cameras.new("Still_Cam"))
        bpy.context.scene.collection.objects.link(obj)
    obj.data.clip_start = 0.02
    obj.data.clip_end = 600.0
    return obj


def _light(name, at, towards, energy, size):
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, bpy.data.lights.new(name, type="AREA"))
        bpy.context.scene.collection.objects.link(obj)
    obj.data.energy = energy
    obj.data.size = size
    obj.data.color = (1.0, 0.94, 0.86)
    obj.matrix_world = look_at(at, towards)
    return obj


def shoot(name: str, eye, target, lens: float, fstop: float = 8.0,
          samples: int = SAMPLES) -> float:
    scene = bpy.context.scene
    cam = _cam()
    cam.data.lens = lens
    cam.matrix_world = look_at(eye, target)
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = fstop
    cam.data.dof.focus_distance = (Vector(eye) - Vector(target)).length
    scene.camera = cam
    scene.cycles.samples = samples
    scene.render.resolution_x, scene.render.resolution_y = RES
    OUT.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(OUT / name)
    scene.render.image_settings.file_format = "PNG"
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    return time.perf_counter() - started


def main() -> dict:
    defects = __import__("json").loads(
        (ROOT / "tank" / "data" / "tank_scene.json").read_text(encoding="utf-8")
    )["defects"]
    visible = next(d for d in defects
                   if d["kind"] == "thinning" and d.get("visible"))
    hidden = next(d for d in defects
                  if d["kind"] == "thinning" and not d.get("visible")
                  and abs(d.get("z_m", 0) - RUN_Z) < 1.2)

    rig = bpy.data.objects.get("Crawler")
    if rig is not None:
        rig.matrix_world = T.crawler_matrix(-0.12, RUN_Z)
    bpy.context.view_layer.update()

    took = {}

    # 1. Establishing. Well back and high, so the curve of the shell is the
    # subject and the crawler is a bright speck on it, which is the true
    # relationship between the machine and the job.
    _light("Still_Flood_A", T.wall_point(-0.45, 1.2, inset=6.5),
           T.wall_point(-0.14, 4.2, inset=0.1), 2600.0, 2.2)
    _light("Still_Flood_B", T.wall_point(0.30, 1.0, inset=7.5),
           T.wall_point(-0.05, 5.0, inset=0.1), 1500.0, 2.0)
    took["establishing"] = shoot(
        "still_establishing.png",
        T.wall_point(0.10, 7.4, inset=9.3),
        T.wall_point(-0.12, 4.0, inset=0.1), 28.0, fstop=11.0)

    # 2. The machine, three quarters and close.
    M = rig.matrix_world if rig is not None else None
    if M is not None:
        eye = tuple(M @ Vector((2.05, -1.25, 0.92)))
        tgt = tuple(M @ Vector((0.16, 0.0, 0.13)))
        _light("Still_Key", tuple(M @ Vector((1.6, -1.5, 1.5))), tgt, 60.0, 0.9)
        _light("Still_Rim", tuple(M @ Vector((-1.2, 1.0, 1.1))), tgt, 34.0, 0.6)
        took["crawler"] = shoot("still_crawler.png", eye, tgt, 55.0, fstop=7.1)

    # 3 and 4. The pair that carries the argument, shot identically so the
    # only difference between them is the wall.
    for name, spot in (("still_corroded.png", visible), ("still_clean.png", hidden)):
        theta, z = spot["theta_rad"], spot["z_m"]
        target = T.wall_point(theta, z, inset=0.0)
        eye = T.wall_point(theta, z + 0.55, inset=1.15)
        _light("Still_Key", T.wall_point(theta - 0.07, z + 1.3, inset=1.9),
               target, 95.0, 0.9)
        _light("Still_Rim", T.wall_point(theta + 0.10, z + 0.4, inset=0.8),
               target, 55.0, 0.45)
        took[name.split(".")[0]] = shoot(name, eye, target, 34.0, fstop=9.0)

    return {"rendered": {k: round(v, 1) for k, v in took.items()},
            "out": str(OUT),
            "visible_patch": visible["id"], "hidden_patch": hidden["id"],
            "note": "the corroded and clean stills are shot identically on "
                    "purpose; the only difference is the wall"}


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=1))
