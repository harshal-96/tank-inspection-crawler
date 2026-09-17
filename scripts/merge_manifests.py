"""Stitch several capture batches into one manifest.

A capture builds one render-only twin per part, because the press error is baked
into the sheet and geometry created after the stage composes never reaches the
renderer. That means the Replicator graph carries one pose sequence per part, of
one entry per frame, so its size grows with parts TIMES frames. At fifty parts and
five views it stops being a graph and becomes a wall: a run that should take
twenty minutes spent thirty in graph construction without writing a single frame.

Batches of ten capture in minutes. This puts them back together so the campaign
sees one run: trials are renumbered end to end and image paths are rewritten
relative to the combined manifest.

    python scripts/merge_manifests.py data/cap50/batch*/manifest.json \\
        --out data/cap50/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_KEYS = ("rgb", "mask", "depth", "normals")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_root = out_path.parent
    combined: dict = {}
    rows: list[dict] = []

    for name in args.manifests:
        path = Path(name)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # Every batch is the same cell photographed by the same camera, so the
        # header is taken from the first and the rest are checked against it
        # rather than silently overwritten.
        if not combined:
            combined = {k: v for k, v in manifest.items() if k != "rows"}
        elif manifest.get("camera") != combined.get("camera"):
            raise SystemExit(f"{path} was captured with a different camera")

        prefix = path.parent.relative_to(out_root).as_posix()
        for row in manifest["rows"]:
            row["trial"] = len(rows)
            row["batch"] = prefix
            row["bead_image"] = f"{prefix}/{row['bead_image']}"
            for view in row.get("views", []):
                for key in IMAGE_KEYS:
                    if key in view:
                        view[key] = f"{prefix}/{view[key]}"
            rows.append(row)

    combined["rows"] = rows
    combined["parts"] = len(rows)
    out_path.write_text(json.dumps(combined, indent=1), encoding="utf-8")

    views = sum(len(r.get("views", [])) for r in rows)
    print(f"merged {len(args.manifests)} batches: {len(rows)} parts, {views} views")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
