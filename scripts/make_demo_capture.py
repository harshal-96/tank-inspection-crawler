"""Cut a full capture down to something a git repository can carry.

The raw captures are about 900 MB for fifty parts, so they are not in this
repository, and for a while the first command in the setup instructions pointed at
a manifest that nobody cloning the repo actually had. Anyone following the README
hit a missing file before they ever saw the cell run.

This writes data/demo/: a handful of parts, every view kept, depth packed to uint16
over the near field and gzipped. The manifest is rewritten to point at the packed
frames and carries the same ground truth as the full one, so the pipeline that runs
on it is the pipeline, not a replay of saved numbers.

What that costs is measured rather than assumed, and the check runs here: each part
is tracked from the raw frame and again from the packed one, and the worst station
shift is printed. It has been under 0.1 mm against a tracked seam accuracy of
0.68 mm.

    python scripts/make_demo_capture.py --parts 6
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.capture import pack_depth, unpack_depth  # noqa: E402
from cell.depth_joint import Intrinsics, backproject, transform_points  # noqa: E402
from cell.seam_track import find_and_track  # noqa: E402

# All PNG already. The mask is what the tracker segments with, and the normals are
# what the figures are shaded from, which is why the pictures in the README can be
# regenerated from what ships rather than only on the machine that captured it. The
# colour pass is 113 KB a view against the normals' 5 KB, and nothing reads it.
COPY_CHANNELS = ("mask", "normals")
DROP_CHANNELS = ("rgb",)


def _stations(depth, intrinsics, to_world, start, end):
    """Track one frame and return the station positions, freeing the cloud after.

    The cloud is a third of a million points and two of them alive at once, per
    part, across a whole capture, is what put this script out of memory the first
    time it ran.
    """
    cloud = transform_points(backproject(depth, intrinsics), to_world)
    track = find_and_track(cloud, start, end, stations=20)
    out = [np.asarray(s.position, dtype=float) for s in track.measured]
    del cloud, track
    gc.collect()
    return out


def worst_station_shift_mm(raw, packed, intrinsics, to_world, start, end) -> float | None:
    """How far the packed frame moves a tracked station. None if either refuses."""
    a = _stations(raw, intrinsics, to_world, start, end)
    b = _stations(packed, intrinsics, to_world, start, end)
    if not a or not b:
        return None
    return max(float(np.linalg.norm(a[i] - b[i])) * 1000.0
               for i in range(min(len(a), len(b))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/cap50/manifest.json")
    parser.add_argument("--out", default="data/demo")
    parser.add_argument("--parts", type=int, default=6)
    parser.add_argument("--keep-beads", action="store_true", default=True,
                        help="Copy the post-weld images, so inspection runs too.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"no capture at {source}. This script runs on the machine that "
              f"captured it; everyone else uses the demo capture it produced.")
        return 1
    root = source.parent
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    manifest = json.loads(source.read_text(encoding="utf-8"))
    camera = manifest["camera"]
    intrinsics = Intrinsics.from_camera(
        camera["width"], camera["height"], camera["focal_length_mm"],
        camera["horizontal_aperture_mm"], camera["vertical_aperture_mm"])
    to_world = np.array(camera["to_world"])
    nominal = geo.nominal_joint()
    floor = geo.TABLE_TOP + geo.PART_T
    start = np.array([nominal[0], nominal[1], floor + 0.04])
    end = np.array([nominal[0], nominal[1], floor + geo.PART_D - 0.02])

    rows, raw_bytes, packed_bytes, shifts = [], 0, 0, []
    print(f"{'part':>5} {'views':>6} {'raw MB':>8} {'packed MB':>10} "
          f"{'worst station shift mm':>23}")
    print("-" * 56)

    for original in manifest["rows"][:args.parts]:
        row = dict(original)
        views, part_raw, part_packed = [], 0, 0
        square_raw_path = square_packed_path = None
        # The squarest view is the one the campaign tracks by default, so it is
        # the one whose packing cost is worth measuring.
        squarest = min(original["views"], key=lambda v: abs(v["orbit_deg"]))["depth"]

        for source_view in original["views"]:
            src = root / source_view["depth"]
            if not src.exists():
                continue
            view = dict(source_view)
            raw = np.load(src)
            packed_rel = f"scans/{Path(source_view['depth']).name}.gz"
            written = pack_depth(raw, out / packed_rel)
            del raw
            gc.collect()
            part_raw += src.stat().st_size
            part_packed += written
            view["depth"] = packed_rel

            for channel in COPY_CHANNELS:
                if channel in view and (root / view[channel]).exists():
                    name = Path(view[channel]).name
                    shutil.copy2(root / view[channel], out / "scans" / name)
                    part_packed += (out / "scans" / name).stat().st_size
                    view[channel] = f"scans/{name}"
            for channel in DROP_CHANNELS:
                view.pop(channel, None)

            if source_view["depth"] == squarest:
                square_raw_path, square_packed_path = src, out / packed_rel
            views.append(view)

        if not views:
            continue
        row["views"] = views

        if args.keep_beads and original.get("bead_image"):
            src_bead = root / original["bead_image"]
            if src_bead.exists():
                name = Path(original["bead_image"]).name
                (out / "beads").mkdir(exist_ok=True)
                shutil.copy2(src_bead, out / "beads" / name)
                part_packed += (out / "beads" / name).stat().st_size
                row["bead_image"] = f"beads/{name}"

        shift = None
        if square_raw_path is not None:
            shift = worst_station_shift_mm(
                np.load(square_raw_path),
                unpack_depth(square_packed_path.read_bytes()),
                intrinsics, to_world, start, end)
            gc.collect()
            if shift is not None:
                shifts.append(shift)

        raw_bytes += part_raw
        packed_bytes += part_packed
        rows.append(row)
        print(f"{row['trial']:>5} {len(views):>6} {part_raw/1e6:>8.1f} "
              f"{part_packed/1e6:>10.2f} "
              f"{'refused' if shift is None else f'{shift:.4f}':>23}")

    manifest["rows"] = rows
    manifest["trials"] = len(rows)
    manifest["capture_dir"] = str(out.resolve())
    manifest["depth_packed"] = True
    manifest["note"] = (
        "A cut-down capture that fits in the repository. Depth is quantised to "
        "uint16 over the near field and gzipped; the full frames are float32. The "
        "worst station shift this causes is printed by scripts/make_demo_capture.py "
        "and has been under 0.1 mm against a tracked seam accuracy of 0.68 mm."
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print("-" * 56)
    print(f"{len(rows)} parts, {raw_bytes/1e6:.0f} MB of depth packed into "
          f"{packed_bytes/1e6:.1f} MB on disk")
    if shifts:
        print(f"worst station shift across them: {max(shifts):.4f} mm")
    print(f"\nwrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
