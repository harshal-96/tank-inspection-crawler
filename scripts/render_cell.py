"""Render the cell from the views worth keeping.

Run it through the running Blender, after building the cell:

    python scripts/blender_send.py code scripts/build_cell_blender.py
    python scripts/blender_send.py code scripts/render_cell.py

The framing is not arbitrary. The corner this cell welds faces out along the same
diagonal the robot stands on, so the natural view of it is the one the arm is in
the way of, and a view square to the tray puts the corner edge on, where it
disappears into the silhouette. These viewpoints sit off that diagonal far enough
that the corner opens up as a vertical edge with a lit face either side of it, and
high enough that the arm comes in from below rather than across it.
"""

import sys

sys.path.insert(0, r"D:\hackathon\nvidia_nebius")

import bpy  # noqa: E402

from scripts.build_cell_blender import SEAM_BOTTOM, SEAM_TOP, aim_camera  # noqa: E402

OUT = r"D:\hackathon\nvidia_nebius\data"
MID = (SEAM_BOTTOM + SEAM_TOP) / 2.0

VIEWS = [
    # Standing behind the cell camera, looking down its own line. This is the view
    # the measurement is taken from, pulled back far enough to see the camera, the
    # arm and the joint at once, so the corner reads as the camera reads it rather
    # than from somewhere the cell has no opinion about.
    ("cell_wide", (1600, 1000),
     (6.512, -2.330, 2.365),
     (4.362, -0.582, 1.030), 40.0),
    # The same line, moved in close on the joint. Same direction on purpose: a
    # close-up from a different angle makes the reader re-orient before they can
    # read it, and there is nothing to be gained by that here.
    ("cell_arc", (1400, 1050),
     (5.774, -1.730, 1.907),
     (4.362, -0.582, 1.020), 48.0),
]


def render():
    scene = bpy.context.scene
    made = []
    for name, (width, height), eye, target, lens in VIEWS:
        scene.render.resolution_x, scene.render.resolution_y = width, height
        aim_camera(eye, target, lens=lens)
        scene.render.filepath = f"{OUT}\\{name}.png"
        bpy.ops.render.render(write_still=True)
        made.append(name)
    return made


result = render()
