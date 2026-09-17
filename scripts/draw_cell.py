"""Draw the cell to scale, from the same constants the simulation builds from.

No renderer involved. This is the model, not a picture of it.

    python scripts/draw_cell.py --out data/cell_model.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402

INK = (28, 32, 38)
MUTED = (130, 140, 152)
CABINET = (188, 198, 210)
CABINET_EDGE = (90, 100, 115)
STOP = (206, 122, 78)
JOINT = (214, 58, 58)
CAMERA = (46, 140, 214)
BEAM = (46, 140, 214, 38)


class Axes:
    """Maps metres onto pixels, y upward."""

    def __init__(self, x0, x1, y0, y1, left, top, width, height):
        self.x0, self.y0 = x0, y0
        self.scale = min(width / (x1 - x0), height / (y1 - y0))
        self.left, self.top, self.height = left, top, height

    def __call__(self, x, y):
        return (
            self.left + (x - self.x0) * self.scale,
            self.top + self.height - (y - self.y0) * self.scale,
        )

    def rect(self, x0, y0, x1, y1):
        a, b = self(x0, y1), self(x1, y0)
        return [a[0], a[1], b[0], b[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/cell_model.png")
    args = parser.parse_args()

    W, H = 1500, 760
    image = Image.new("RGB", (W, H), (250, 251, 253))
    draw = ImageDraw.Draw(image, "RGBA")

    settled_x = geo.FIXTURE_X + 0.01
    half_w, half_h = geo.PART_W / 2, geo.PART_H / 2
    joint = geo.nominal_joint()

    # ----------------------------------------------------------- plan view
    draw.text((40, 24), "PLAN  (looking down)", fill=INK)
    plan = Axes(3.30, 5.40, -1.55, 0.95, 40, 60, 700, 640)

    for metres in (3.5, 4.0, 4.5, 5.0):
        a, b = plan(metres, -1.55), plan(metres, 0.95)
        draw.line([a, b], fill=(232, 236, 241))
        draw.text((a[0] - 12, a[1] + 4), f"x={metres}", fill=MUTED)

    draw.rectangle(plan.rect(3.15, -0.85, 4.75, 0.85), outline=(205, 212, 220))
    draw.text(plan(3.18, 0.88), "fixture table", fill=MUTED)

    draw.rectangle(plan.rect(4.41, -0.65, 4.46, 0.65), fill=STOP)
    draw.text(plan(4.48, 0.50), "end stop", fill=STOP)

    # The cabinet, hollow, drawn with its wall thickness.
    draw.rectangle(
        plan.rect(settled_x - half_w, -half_h, settled_x + half_w, half_h),
        fill=CABINET, outline=CABINET_EDGE, width=2,
    )
    draw.rectangle(
        plan.rect(
            settled_x - half_w + geo.PART_T, -half_h + geo.PART_T,
            settled_x + half_w - geo.PART_T, half_h - geo.PART_T,
        ),
        fill=(250, 251, 253), outline=CABINET_EDGE,
    )
    draw.text(plan(settled_x - 0.30, 0.10), "cabinet", fill=CABINET_EDGE)
    draw.text(plan(settled_x - 0.30, -0.02), f"{geo.PART_W} x {geo.PART_H} m", fill=MUTED)
    draw.text(plan(settled_x - 0.30, -0.14), f"wall {geo.PART_T * 1000:.0f} mm", fill=MUTED)

    # Camera and its line of sight to the joint.
    eye = plan(geo.CAMERA_EYE[0], geo.CAMERA_EYE[1])
    target = plan(*joint)
    draw.polygon([eye, plan(joint[0] - 0.26, joint[1] + 0.26), plan(joint[0] + 0.20, joint[1] - 0.20)],
                 fill=BEAM)
    draw.line([eye, target], fill=CAMERA, width=2)
    draw.ellipse([eye[0] - 7, eye[1] - 7, eye[0] + 7, eye[1] + 7], fill=CAMERA)
    draw.text((eye[0] - 26, eye[1] + 12), "depth camera", fill=CAMERA)
    draw.text((eye[0] - 26, eye[1] + 24), f"{0.95:.2f} m standoff", fill=MUTED)

    draw.ellipse([target[0] - 6, target[1] - 6, target[0] + 6, target[1] + 6], fill=JOINT)
    draw.text((target[0] + 10, target[1] - 26), "WELD JOINT", fill=JOINT)
    draw.text((target[0] + 10, target[1] - 14), f"({joint[0]:.2f}, {joint[1]:.2f})", fill=MUTED)

    draw.line([plan(0.8, -0.66), plan(4.3, -0.66)], fill=(200, 208, 216))
    draw.line([plan(0.8, 0.66), plan(4.3, 0.66)], fill=(200, 208, 216))
    draw.text(plan(3.35, -0.78), "guide rails (invisible to camera)", fill=MUTED)

    # ------------------------------------------------------ elevation view
    draw.text((800, 24), "ELEVATION  (looking along the line)", fill=INK)
    elev = Axes(-0.95, 0.95, 0.60, 1.32, 880, 110, 520, 520)

    draw.rectangle(elev.rect(-0.95, 0.67, 0.95, 0.75), fill=(224, 229, 235))
    draw.text(elev(-0.92, 0.77), f"table top  z={geo.TABLE_TOP}", fill=MUTED)

    top_z = geo.TABLE_TOP + geo.PART_D
    draw.rectangle(elev.rect(-half_h, geo.TABLE_TOP, half_h, top_z),
                   fill=CABINET, outline=CABINET_EDGE, width=2)
    draw.text(elev(-0.30, 1.05), f"cabinet, {geo.PART_D} m tall", fill=CABINET_EDGE)

    # The joint edge runs the full height of the cabinet.
    a, b = elev(-half_h, geo.TABLE_TOP), elev(-half_h, top_z)
    draw.line([a, b], fill=JOINT, width=6)
    draw.text((a[0] - 108, (a[1] + b[1]) / 2 - 16), "WELD JOINT", fill=JOINT)
    draw.text((a[0] - 108, (a[1] + b[1]) / 2 - 2), f"runs {geo.PART_D * 1000:.0f} mm", fill=MUTED)
    draw.text((a[0] - 108, (a[1] + b[1]) / 2 + 10), "top to bottom", fill=MUTED)

    cam = elev(-half_h - 0.30, geo.CAMERA_EYE[2])
    draw.line([cam, elev(-half_h, geo.CAMERA_AIM[2])], fill=CAMERA, width=2)
    draw.ellipse([cam[0] - 7, cam[1] - 7, cam[0] + 7, cam[1] + 7], fill=CAMERA)
    draw.text((cam[0] - 30, cam[1] - 24), f"camera z={geo.CAMERA_EYE[2]:.2f}", fill=CAMERA)

    draw.text(
        (40, H - 34),
        "The cell measures the joint from DEPTH only: a plane fitted to each "
        "panel face, then intersected. No labels, no training, nothing sim-only.",
        fill=INK,
    )
    draw.text(
        (40, H - 20),
        f"Bead {geo.BEAD_SIZE[0] * 1000:.0f} mm, sitting {geo.BEAD_PROUD * 1000:.0f} mm "
        f"outboard of the corner, ripple pitch {geo.RIPPLE_PITCH * 1000:.1f} mm.",
        fill=MUTED,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
