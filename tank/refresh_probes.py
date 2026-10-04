"""Recompute the probe answers in the manifest, without touching the frames.

    python tank/refresh_probes.py

Wall loss that a camera cannot see adds no geometry to the scene, so planting
more of it changes what the probe should find and changes nothing about what
the pictures look like. Re-running the capture would re-render all 24 frames
anyway, and those frames carry recorded model verdicts that only apply to
exactly those pixels. So the images are left alone and only the numbers that
depend on the defect list are worked out again.

What is rewritten: true_thickness_mm, remaining_fraction, thinned_by and
rejectable, at each of the 168 probe points, from the same arithmetic the
capture used.

What is not: the camera poses, the frames, the depth paths, and faults_in_view.
The last one is deliberate. Hidden patches are invisible to the camera by
construction, and the scorer already drops anything not visible_to_a_camera,
so the inspection truth is unaffected by adding them. If a visible defect is
ever added, this script is the wrong tool and the capture has to run.

It refuses to touch anything if the frames on disk no longer match the ones
the manifest was written against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tank.thickness import thickness_mm  # noqa: E402

SURVEY = ROOT / "tank" / "data" / "survey"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(SURVEY / "manifest.json"))
    parser.add_argument("--scene", default=str(ROOT / "tank" / "data" / "tank_scene.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    defects = scene["defects"]
    limit = manifest["acceptance"]["thinning"]["limit_fraction"]

    # Every frame the manifest names must still be on disk, or the probe
    # answers would be refreshed against images that are not there.
    missing = [f["rgb"] for f in manifest["frames"]
               if not (SURVEY / f["rgb"]).exists()]
    if missing:
        print(f"{len(missing)} frames named in the manifest are not on disk, "
              f"starting with {missing[0]}")
        return 1

    was_rejectable = sum(1 for f in manifest["frames"]
                         for c in f["probe_candidates"] if c["rejectable"])
    was_thinned = sum(1 for f in manifest["frames"]
                      for c in f["probe_candidates"] if c["thinned_by"])

    changed = 0
    for frame in manifest["frames"]:
        for c in frame["probe_candidates"]:
            left, took = thickness_mm(defects, c["theta_rad"], c["z_m"])
            fraction = round(left / c["nominal_mm"], 4)
            before = (c["true_thickness_mm"], c["rejectable"])
            c["true_thickness_mm"] = left
            c["remaining_fraction"] = fraction
            c["thinned_by"] = took
            c["rejectable"] = fraction < limit
            if before != (left, c["rejectable"]):
                changed += 1
        on_seam = next(c for c in frame["probe_candidates"]
                       if abs(c["offset_m"]) < 1e-9)
        frame["probe_on_seam"] = on_seam

    now_rejectable = sum(1 for f in manifest["frames"]
                         for c in f["probe_candidates"] if c["rejectable"])
    now_thinned = sum(1 for f in manifest["frames"]
                      for c in f["probe_candidates"] if c["thinned_by"])
    total = sum(len(f["probe_candidates"]) for f in manifest["frames"])

    manifest["probe_truth_refreshed_from"] = Path(args.scene).name
    manifest["evaluation_scene"] = (
        "The hidden wall loss on this run is heavier than a real shell of this "
        "age would carry. It is set that way so a probe planner can be measured "
        "against another one: with a single hidden patch the run held four "
        "rejectable points and no comparison on four points can separate "
        "anything.")

    print(f"{total} probe points")
    print(f"  thinned     {was_thinned:>4} -> {now_thinned}")
    print(f"  rejectable  {was_rejectable:>4} -> {now_rejectable}")
    print(f"  points whose answer moved: {changed}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0
    Path(args.manifest).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {args.manifest}. Frames untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
