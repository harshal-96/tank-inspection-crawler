"""Run the seam tracker on a real depth capture and draw it on the part.

Everything here comes out of the depth buffer: the point cloud is the depth image
back-projected, the seam is measured from that cloud, and the picture underneath
is shaded from depth and normals. No colour pass is involved at any stage.

    python scripts/track_on_model.py --manifest data/cap50/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.capture import load_depth  # noqa: E402
from cell.depth_joint import Intrinsics, backproject, project, transform_points  # noqa: E402
from cell.seam_track import TOUCHING_TOLERANCE_MM, find_and_track  # noqa: E402
from scripts.shade_from_buffers import decode_normals, shade  # noqa: E402

TAUGHT = (150, 155, 165)
MEASURED = (36, 140, 214)
TOUCHING = (52, 170, 96)
OPEN = (222, 96, 36)
TRUTH = (214, 58, 58)
INK = (24, 28, 34)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cap50/manifest.json")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--view", type=float, default=0.0)
    parser.add_argument("--stations", type=int, default=20)
    parser.add_argument("--out", default="data/seam_on_model.png")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    camera = manifest["camera"]
    row = next(r for r in manifest["rows"] if r["trial"] == args.part)
    if "forming" not in row or "true_seam" not in row:
        raise SystemExit(
            f"{manifest_path} was captured before the press forming model existed, so "
            "it carries no ground truth to draw against. Re-capture with "
            "cell/sim/weld_station.py."
        )
    view = min(row["views"], key=lambda v: abs(v["orbit_deg"] - args.view))

    intrinsics = Intrinsics.from_camera(
        camera["width"],
        camera["height"],
        camera["focal_length_mm"],
        camera["horizontal_aperture_mm"],
        camera["vertical_aperture_mm"],
    )
    to_world = np.array(camera["to_world"])

    depth = load_depth(root / view["depth"])
    normals, written = decode_normals(root / view["normals"])

    # The point cloud IS the depth image, back-projected. Nothing else feeds this.
    cloud = transform_points(backproject(depth, intrinsics), to_world)

    # The taught seam: the corner of the part where a fixed program was told the
    # part would be. It starts clear of the base, which occludes the bottom of the
    # corner from any camera standing outside the part.
    nominal = geo.nominal_joint()
    floor = geo.TABLE_TOP + geo.PART_T
    seam_start = np.array([nominal[0], nominal[1], floor + 0.04])
    seam_end = np.array([nominal[0], nominal[1], floor + geo.PART_D - 0.02])

    track = find_and_track(cloud, seam_start, seam_end, stations=args.stations)
    summary = track.summary()

    canvas = Image.fromarray(shade(normals, written, depth)).convert("RGB")
    draw = ImageDraw.Draw(canvas)

    # The taught path, projected into the picture.
    taught_uv, _ = project(np.vstack([seam_start, seam_end]), intrinsics, to_world)
    draw.line([tuple(taught_uv[0]), tuple(taught_uv[1])], fill=TAUGHT, width=3)

    # Where the seam really is, which only the simulation knows. Drawn so the
    # measurement can be judged against the answer rather than against itself.
    truth = np.array(row.get("true_seam", []))
    if len(truth):
        truth = truth[truth[:, 2] >= seam_start[2]]
    if len(truth) > 1:
        truth_uv, _ = project(truth, intrinsics, to_world)
        draw.line([tuple(p) for p in truth_uv], fill=TRUTH, width=3)

    measured = track.measured
    if measured:
        points = np.array([s.position for s in measured])
        uv, _ = project(points, intrinsics, to_world)
        draw.line([tuple(p) for p in uv], fill=MEASURED, width=2)
        for station, pixel in zip(measured, uv, strict=True):
            colour = TOUCHING if station.touching else OPEN
            draw.ellipse(
                [pixel[0] - 4, pixel[1] - 4, pixel[0] + 4, pixel[1] + 4],
                fill=colour,
                outline=(255, 255, 255),
            )

    lines = [
        f"depth points used: {len(cloud):,}",
        f"stations measured: {summary.get('stations_measured', 0)} of {summary.get('of', 0)}",
        f"seam wander off the taught line: {summary.get('path_wander_mm', 0):.2f} mm",
        f"gap mean {summary.get('gap_mm_mean', 0):.2f} mm, worst "
        f"{summary.get('gap_mm_max', 0):.2f} mm",
        f"stations touching (under {TOUCHING_TOLERANCE_MM} mm): "
        f"{summary.get('stations_touching', 0)}",
        f"press left: {row['forming']['right_lean_deg']:+.2f}/"
        f"{row['forming']['front_lean_deg']:+.2f} deg lean, "
        f"{row['forming']['right_bow_mm']:+.1f}/{row['forming']['front_bow_mm']:+.1f} mm bow",
        f"taught path misses the real seam by {row['taught_miss_mm']:.1f} mm",
    ]
    box_height = 14 * len(lines) + 12
    draw.rectangle([0, canvas.height - box_height, canvas.width, canvas.height],
                   fill=(252, 252, 253))
    for i, line in enumerate(lines):
        draw.text((8, canvas.height - box_height + 6 + i * 14), line, fill=INK)

    draw.text((8, 8), "grey: taught path", fill=TAUGHT)
    draw.text((8, 22), "red: where the seam really is", fill=TRUTH)
    draw.text((8, 36), "blue: seam measured from depth", fill=MEASURED)
    draw.text((8, 50), "green: edges touching   orange: gap open", fill=OPEN)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
