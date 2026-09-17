"""Draw the part: an open-topped tray folded from one notched blank.

Four corner notches are cut in the flat blank so the four sides fold up without
fouling each other. Once folded, each pair of neighbouring sides meets along a
vertical corner line, and that line is the seam the robot welds. Nothing overlaps
and nothing is let into a face: two folded edges come together, which is what has
to be found and followed.

    python scripts/draw_part.py --out data/part.png
    python scripts/draw_part.py --blank --out data/blank.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Millimetres, so the numbers read the way a drawing does.
BASE_W = 900.0
BASE_H = 560.0
WALL_D = 185.0
THICK = 6.0

OUTLINE = (91, 127, 166)
FOLD = (120, 150, 180)
SEAM = (59, 155, 232)
BLANK_FILL = (216, 225, 235)
BLANK_INNER = (206, 217, 229)

TOP = (243, 246, 249)      # the rim of the wall, seen from above
OUTER = (232, 237, 242)    # outer face of a near wall
OUTER_DIM = (218, 226, 234)
INNER = (184, 198, 213)    # inner face of a far wall
FLOOR = (143, 169, 192)    # the base, seen from inside the tray
EDGE = (120, 145, 172)

COS30 = math.cos(math.radians(30.0))
SIN30 = math.sin(math.radians(30.0))


def iso(point, scale, origin):
    """Project a millimetre point into the picture."""
    x, y, z = point
    sx = (x - y) * COS30
    sy = (x + y) * SIN30 - z
    return (origin[0] + sx * scale, origin[1] + sy * scale)


def depth(face):
    """How near the viewer a face sits. The viewer is up the (1, 1, 1) diagonal."""
    return sum(sum(v) for v in face) / len(face)


def box(x0, x1, y0, y1, z0, z1, top, side_x, side_y):
    """The three faces of a box that a viewer up the (1, 1, 1) diagonal can see."""
    return [
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top),
        ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], side_x),
        ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], side_y),
    ]


def dashed(draw, a, b, colour, dash=9, gap=7):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    travelled = 0.0
    while travelled < length:
        end = min(travelled + dash, length)
        draw.line(
            [(a[0] + ux * travelled, a[1] + uy * travelled),
             (a[0] + ux * end, a[1] + uy * end)],
            fill=colour, width=2,
        )
        travelled = end + gap


def draw_blank(draw, scale, origin, hw, hh, d):
    """The flat blank: a rectangle with a square notch taken out of each corner."""
    outline = [
        (-hw, -hh - d), (hw, -hh - d), (hw, -hh), (hw + d, -hh),
        (hw + d, hh), (hw, hh), (hw, hh + d), (-hw, hh + d),
        (-hw, hh), (-hw - d, hh), (-hw - d, -hh), (-hw, -hh),
    ]
    poly = [iso((x, y, 0.0), scale, origin) for x, y in outline]
    draw.polygon(poly, fill=BLANK_FILL, outline=OUTLINE)
    draw.line(poly + [poly[0]], fill=OUTLINE, width=2)

    # The fold lines. The metal bends up along these four.
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    for i in range(4):
        a = iso((*corners[i], 0.0), scale, origin)
        b = iso((*corners[(i + 1) % 4], 0.0), scale, origin)
        dashed(draw, a, b, FOLD)

    # The eight edges that pair up at the four corners once the sides are folded.
    # Each pair becomes one vertical seam.
    pairs = [
        (((hw, hh), (hw + d, hh)), ((hw, hh), (hw, hh + d))),
        (((hw, -hh), (hw + d, -hh)), ((hw, -hh), (hw, -hh - d))),
        (((-hw, hh), (-hw - d, hh)), ((-hw, hh), (-hw, hh + d))),
        (((-hw, -hh), (-hw - d, -hh)), ((-hw, -hh), (-hw, -hh - d))),
    ]
    for pair in pairs:
        for a, b in pair:
            draw.line([iso((*a, 0.0), scale, origin), iso((*b, 0.0), scale, origin)],
                      fill=SEAM, width=4)


def draw_tray(draw, scale, origin, hw, hh, d, t):
    """The same blank folded up. Sides rise from the fold lines, corners meet."""
    faces = []
    faces += box(-hw, hw, -hh, hh, 0.0, t, FLOOR, OUTER_DIM, OUTER_DIM)

    # Each wall stands on its fold line and leans outward by the sheet thickness,
    # so the inside of the tray keeps the base dimensions.
    walls = [
        (hw, hw + t, -hh, hh, OUTER, OUTER),          # near wall, faces down-right
        (-hw - t, -hw, -hh, hh, INNER, OUTER_DIM),    # far wall, we see inside it
        (-hw, hw, hh, hh + t, OUTER_DIM, OUTER),      # near wall, faces down-left
        (-hw, hw, -hh - t, -hh, OUTER_DIM, INNER),    # far wall, we see inside it
    ]
    for x0, x1, y0, y1, side_x, side_y in walls:
        faces += box(x0, x1, y0, y1, 0.0, t + d, TOP, side_x, side_y)

    faces.sort(key=lambda f: depth(f[0]))
    for verts, colour in faces:
        poly = [iso(v, scale, origin) for v in verts]
        draw.polygon(poly, fill=colour, outline=EDGE)

    # The four corner seams, drawn where the folded edges come together.
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * (hw + t / 2), sy * (hh + t / 2)
            draw.line([iso((x, y, 0.0), scale, origin),
                       iso((x, y, t + d), scale, origin)],
                      fill=EDGE, width=2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=BASE_W)
    parser.add_argument("--height", type=float, default=BASE_H)
    parser.add_argument("--depth", type=float, default=WALL_D)
    parser.add_argument("--thickness", type=float, default=THICK)
    parser.add_argument("--blank", action="store_true",
                        help="Draw the flat blank the tray is folded from instead.")
    parser.add_argument("--out", default="data/part.png")
    args = parser.parse_args()

    hw, hh = args.width / 2.0, args.height / 2.0
    d, t = args.depth, args.thickness

    W, H = 1200, 900
    image = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    if args.blank:
        draw_blank(draw, 0.52, (W / 2, H / 2), hw, hh, d)
    else:
        draw_tray(draw, 0.62, (W / 2, H / 2 + 120), hw, hh, d, t)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    if args.blank:
        print(f"blank {args.width + 2 * d:.0f} x {args.height + 2 * d:.0f} mm, "
              f"{d:.0f} mm flanges, {t:.0f} mm sheet, four corner notches")
    else:
        print(f"tray {args.width:.0f} x {args.height:.0f} x {d:.0f} mm inside, "
              f"{t:.0f} mm sheet")
        print("four corner seams, each a vertical line where two folded edges meet")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
