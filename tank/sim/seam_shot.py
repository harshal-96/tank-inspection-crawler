"""Ten seconds of the crawler working a seam, set up and saved ready to render.

    python scripts/blender_send.py tank/sim/blender_tank.py
    python scripts/blender_send.py tank/sim/crawler_detail.py
    python scripts/blender_send.py tank/sim/look.py
    python scripts/blender_send.py tank/sim/seam_shot.py      # keyframes and saves

The last step writes tank/data/shots/seam_shot.blend and does not render. The
render runs headless afterwards so Blender's socket is not held open for hours
and the machine stays usable.

The shot. The crawler starts behind the undercut and drives 1.5 metres along
the C2 seam at 0.15 m/s, which is roughly what a magnetic crawler actually
does. On the way it passes over a groove at the weld toe and then out across
the corroded patch, finishing near the porosity. The camera starts wide enough
to show it is inside a tank and eases in to three quarters as the rust fills
the frame, so the move ends on the thing the machine is there to find.

Everything the camera and the lights do is defined in the robot's own frame and
baked to world per frame, which means the rig travels with the crawler and the
exposure does not drift as it drives into the dark. Set up in world space the
shot lights the start of the run and loses the end of it.

Nothing here touches the dataset.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tank.sim import blender_tank as T  # noqa: E402
from tank.sim.meshkit import look_at  # noqa: E402

FPS = 24
SECONDS = 10
FRAMES = FPS * SECONDS

# Where it drives. Starts clear of the groove so the groove is something it
# arrives at rather than something it is already sitting on.
THETA_FROM = -0.165
THETA_TO = -0.065
RUN_Z = 4.0
THETA_MID = (THETA_FROM + THETA_TO) / 2

# The camera, in the robot's own frame: behind, off to one side, above. It
# closes in over the ten seconds rather than cutting, because one unbroken
# move is more convincing than three clever ones.
# The camera is locked in world space, not carried by the robot, and the robot
# drives across it. Two attempts at a camera mounted in the robot's own frame
# both failed the same way: ride along with it and the machine sits still in
# the middle of the picture while the wall slides past, which is the one thing
# that reads as a render. Nail the camera to the shell and let the crawler
# cross the frame and the motion becomes the subject.
#
# Placed above the seam and out from the wall, so the bead runs horizontally
# across the frame and the whole 1.5 metres of travel fits inside it.
CAM_ABOVE_M = 1.15        # up the shell from the seam
CAM_OUT_START = 2.60      # out from the plate, where it begins
CAM_OUT_END = 2.20        # and where the slow push-in ends
LENS_START = 28.0
LENS_END = 32.0

# Key, rim and fill, also in the robot's frame so they travel with it.
# Fixed in world as well, placed around the middle of the run so the whole
# travelled band stays lit. Carried on the robot they light it beautifully and
# drag a moving pool of light across the wall behind it, which no work lamp
# rigged on a tank floor has ever done.
# (arc offset along the seam, height above it, distance out from the plate,
#  watts, size)
HERO_LIGHTS = (
    ("Hero_Key", (-0.70, 1.45, 1.70), 120.0, 1.0),
    ("Hero_Rim", (1.15, 0.40, 0.70), 95.0, 0.45),
    ("Hero_Fill", (0.10, -0.95, 1.50), 12.0, 1.5),
)

SAMPLES = 192
RES = (1920, 1080)
OUT_BLEND = ROOT / "tank" / "data" / "shots" / "seam_shot.blend"
OUT_VIDEO = ROOT / "tank" / "data" / "shots" / "seam_shot.mp4"
FRAME_DIR = ROOT / "tank" / "data" / "shots" / "seam_frames"


def ease(t: float) -> float:
    """Smoothstep. A move that starts and stops abruptly reads as a machine
    cut, and this one is meant to read as somebody operating a camera."""
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def _key(obj, frame):
    obj.keyframe_insert(data_path="location", frame=frame)
    obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def _light(name, energy, size):
    obj = bpy.data.objects.get(name)
    if obj is None or obj.type != "LIGHT":
        data = bpy.data.lights.new(name, type="AREA")
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
    obj.data.energy = energy
    obj.data.size = size
    obj.data.color = (1.0, 0.94, 0.86)
    obj.rotation_mode = "XYZ"
    return obj


def main() -> dict:
    scene = bpy.context.scene
    rig = bpy.data.objects.get("Crawler")
    if rig is None:
        return {"error": "no Crawler in the scene; run blender_tank.py first"}

    rig.rotation_mode = "XYZ"
    for obj in (rig,):
        obj.animation_data_clear()

    cam_obj = bpy.data.objects.get("Hero_Cam")
    if cam_obj is None:
        cam_obj = bpy.data.objects.new("Hero_Cam", bpy.data.cameras.new("Hero_Cam"))
        scene.collection.objects.link(cam_obj)
    cam_obj.rotation_mode = "XYZ"
    cam_obj.animation_data_clear()
    cam = cam_obj.data
    cam.clip_start = 0.02
    cam.clip_end = 400.0
    cam.dof.use_dof = True
    cam.dof.aperture_fstop = 9.0
    cam.animation_data_clear()

    lights = []
    for name, place, energy, size in HERO_LIGHTS:
        obj = _light(name, energy, size)
        obj.animation_data_clear()
        lights.append((obj, place))

    scene.frame_start, scene.frame_end = 1, FRAMES
    scene.render.fps = FPS

    # Linear between baked keys. The easing is already in the positions, so
    # letting Blender lay its own curve over the top makes the move float.
    # Set before inserting rather than fixed afterwards, because Blender 5
    # moved actions to slots and there is no longer an action.fcurves to walk.
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"
    except Exception:
        pass

    for frame in range(1, FRAMES + 1):
        t = ease((frame - 1) / (FRAMES - 1))

        theta = lerp(THETA_FROM, THETA_TO, t)
        rig.matrix_world = T.crawler_matrix(theta, RUN_Z)
        _key(rig, frame)

        # The camera stays where it is and watches the middle of the run.
        target = Vector(T.wall_point(THETA_MID, RUN_Z, inset=0.0))
        eye = Vector(T.wall_point(THETA_MID, RUN_Z + CAM_ABOVE_M,
                                  inset=lerp(CAM_OUT_START, CAM_OUT_END, t)))
        cam_obj.matrix_world = look_at(tuple(eye), tuple(target))
        _key(cam_obj, frame)

        cam.lens = lerp(LENS_START, LENS_END, t)
        cam.keyframe_insert(data_path="lens", frame=frame)
        # Focus on the wall, not on the camera's distance to the robot, so the
        # seam stays sharp for the whole pass.
        cam.dof.focus_distance = (eye - target).length
        cam.dof.keyframe_insert(data_path="focus_distance", frame=frame)

        for obj, (arc, up, out, *_rest) in lights:
            obj.matrix_world = look_at(
                T.wall_point(THETA_MID + arc / T.RADIUS, RUN_Z + up, inset=out),
                tuple(target))
            _key(obj, frame)

    scene.camera = cam_obj
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.4

    # This Blender is built without video output, so the shot lands as a PNG
    # sequence and is encoded afterwards. That is the better way round anyway:
    # a dropped render can be resumed, and the frames survive a bad encode.
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.compression = 15
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(FRAME_DIR / "f_")

    OUT_BLEND.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT_BLEND), copy=True)

    arc = (THETA_TO - THETA_FROM) * T.RADIUS
    return {
        "frames": FRAMES,
        "fps": FPS,
        "travel_m": round(arc, 3),
        "speed_m_s": round(arc / SECONDS, 3),
        "passes_over": [d["id"] for d in
                        __import__("json").loads(
                            (ROOT / "tank" / "data" / "tank_scene.json")
                            .read_text(encoding="utf-8"))["defects"]
                        if d.get("z_m", 99) < 4.6 and d.get("z_m", -9) > 3.4
                        and THETA_FROM - 0.02 <= d.get("theta_rad", 9) <= THETA_TO + 0.02],
        "samples": SAMPLES,
        "resolution": list(RES),
        "blend": str(OUT_BLEND),
        "frames_dir": str(FRAME_DIR),
        "video": str(OUT_VIDEO),
        "binary": bpy.app.binary_path,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=1))
