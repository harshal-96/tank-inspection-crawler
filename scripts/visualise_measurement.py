"""Draw what one measurement actually sees.

Takes a captured view and overlays the working window, the two panel faces the
fitter claimed, the joint it derived from them, and where the joint truly is. Good
for checking a campaign by eye, and for showing someone what the cell is doing.

    python scripts/visualise_measurement.py --part 0 --view 0.0 --out data/look.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.depth_joint import (  # noqa: E402
    backproject,
    fit_plane_ransac,
    intersect_planes,
    project,
    transform_points,
    working_volume,
)
from scripts.run_campaign import (  # noqa: E402
    JOINT_DIRECTION,
    RANGE_MARGIN_M,
    ROI_HALF_HEIGHT_PX,
    ROI_HALF_WIDTH_PX,
    load_context,
)

FACE_A_TINT = (90, 190, 255)
FACE_B_TINT = (255, 170, 60)
DETECTED = (60, 230, 120)
TRUTH = (255, 70, 70)


def tint(image: np.ndarray, pixels: tuple[np.ndarray, np.ndarray], colour, strength=0.45):
    rows, cols = pixels
    patch = image[rows, cols].astype(float)
    image[rows, cols] = (patch * (1 - strength) + np.array(colour) * strength).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/captures/manifest.json")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--view", type=float, default=0.0)
    parser.add_argument("--out", default="data/look.png")
    args = parser.parse_args()

    manifest, ctx = load_context(Path(args.manifest))
    row = next(r for r in manifest["rows"] if r["trial"] == args.part)
    view = next(v for v in row["views"] if abs(v["orbit_deg"] - args.view) < 1e-6)

    depth = np.load(ctx.root / view["depth"])
    mask = working_volume(
        depth,
        ctx.intrinsics,
        ctx.roi_centre,
        ROI_HALF_WIDTH_PX,
        ROI_HALF_HEIGHT_PX,
        ctx.expected_range - RANGE_MARGIN_M,
        ctx.expected_range + RANGE_MARGIN_M,
    )

    # Repeat the fit, keeping hold of which pixel went into which face.
    valid = np.isfinite(depth) & (depth > 0) & mask
    rows, cols = np.nonzero(valid)
    cloud = transform_points(backproject(depth, ctx.intrinsics, mask), ctx.to_world)
    rng = np.random.default_rng(0)
    face_a, inliers_a = fit_plane_ransac(cloud, 0.002, 200, rng)
    rest = ~inliers_a
    face_b, inliers_b = fit_plane_ransac(cloud[rest], 0.002, 200, rng)
    point, direction = intersect_planes(face_a, face_b)
    if direction @ JOINT_DIRECTION < 0:
        direction = -direction

    canvas = np.asarray(Image.open(ctx.root / view["rgb"]).convert("RGB")).copy()
    tint(canvas, (rows[inliers_a], cols[inliers_a]), FACE_A_TINT)
    b_rows, b_cols = rows[rest][inliers_b], cols[rest][inliers_b]
    tint(canvas, (b_rows, b_cols), FACE_B_TINT)

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    draw.rectangle(
        [
            ctx.roi_centre[0] - ROI_HALF_WIDTH_PX,
            ctx.roi_centre[1] - ROI_HALF_HEIGHT_PX,
            ctx.roi_centre[0] + ROI_HALF_WIDTH_PX,
            ctx.roi_centre[1] + ROI_HALF_HEIGHT_PX,
        ],
        outline=(255, 255, 255),
        width=1,
    )

    def draw_line(xy: tuple[float, float], colour, width):
        ends = np.array(
            [[xy[0], xy[1], geo.TABLE_TOP + 0.02], [xy[0], xy[1], geo.TABLE_TOP + geo.PART_D]]
        )
        uv, _ = project(ends, ctx.intrinsics, ctx.to_world)
        draw.line([tuple(uv[0]), tuple(uv[1])], fill=colour, width=width)

    truth = tuple(row["true_joint"])
    scale = (ctx.joint_height - point[2]) / direction[2]
    found = (point + scale * direction)[:2]

    draw_line(truth, TRUTH, 3)
    draw_line((found[0], found[1]), DETECTED, 2)

    error_mm = 1000.0 * float(np.linalg.norm(found - np.array(truth)))
    balance = 2.0 * min(inliers_a.sum(), inliers_b.sum()) / (inliers_a.sum() + inliers_b.sum())
    caption = (
        f"part {args.part}  view {args.view:+.1f} deg   "
        f"face A {inliers_a.sum()} px   face B {inliers_b.sum()} px   "
        f"balance {balance:.3f}   error {error_mm:.2f} mm"
    )
    draw.rectangle([0, image.height - 18, image.width, image.height], fill=(16, 18, 22))
    draw.text((6, image.height - 14), caption, fill=(230, 235, 240))
    draw.text((6, 6), "white: search window", fill=(255, 255, 255))
    draw.text((6, 18), "blue / orange: the two faces fitted", fill=(150, 200, 255))
    draw.text((6, 30), "green: joint found    red: true joint", fill=(150, 255, 190))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print(f"{caption}\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
