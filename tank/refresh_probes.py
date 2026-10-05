"""Recompute the answer key in the manifest, without touching the frames.

    python tank/refresh_probes.py            # rewrite the key, if the frames still match
    python tank/refresh_probes.py --check    # report what would change, write nothing

The frames carry recorded model verdicts that apply to exactly those pixels, so
they are never re-rendered to change an answer. Everything the answers depend
on is worked out again instead, from the scene file and from code that needs
no Blender:

- faults_in_view at every station, through tank/truth.py, which projects each
  fault through the crawler's camera at the depth it was drawn;
- true_thickness_mm, remaining_fraction, thinned_by and rejectable at each of
  the 168 probe points, through tank/thickness.py;
- the manifest's own copy of the defect list, from tank_scene.json.

Wall loss a camera cannot see adds no geometry, so planting more of it changes
the probe answers and not one pixel. A visible fault is different: adding one
changes the pictures, and then the capture has to run again. This script checks
for that. It refuses to write anything unless every frame on disk matches the
digest in frames.sha256, which is the record of the pixels the verdicts were
given against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tank.thickness import thickness_mm  # noqa: E402
from tank.truth import faults_in_view, truth_for  # noqa: E402

SURVEY = ROOT / "tank" / "data" / "survey"
DIGESTS = SURVEY / "frames.sha256"


def frames_match(survey: Path, frames: list[dict]) -> list[str]:
    """Every frame whose bytes are not the ones recorded, or that is missing."""
    if not (survey / DIGESTS.name).exists():
        return [f"{DIGESTS.name} is missing"]
    recorded = {}
    for line in (survey / DIGESTS.name).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            recorded[name.strip().lstrip("*")] = digest
    wrong = []
    for frame in frames:
        path = survey / frame["rgb"]
        if not path.exists():
            wrong.append(f"{frame['rgb']} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != recorded.get(frame["rgb"]):
            wrong.append(f"{frame['rgb']} does not match {DIGESTS.name}")
    return wrong


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(SURVEY / "manifest.json"))
    parser.add_argument("--scene", default=str(ROOT / "tank" / "data" / "tank_scene.json"))
    parser.add_argument("--check", "--dry-run", dest="check", action="store_true",
                        help="report what would change and write nothing")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene = json.loads(Path(args.scene).read_text(encoding="utf-8"))
    defects = scene["defects"]
    limit = manifest["acceptance"]["thinning"]["limit_fraction"]

    wrong = frames_match(manifest_path.parent, manifest["frames"])
    if wrong:
        print(f"refusing: {len(wrong)} frame(s) are not the recorded pixels, "
              f"starting with {wrong[0]}")
        return 1

    was_rejectable = sum(1 for f in manifest["frames"]
                         for c in f["probe_candidates"] if c["rejectable"])
    was_thinned = sum(1 for f in manifest["frames"]
                      for c in f["probe_candidates"] if c["thinned_by"])

    moved_points = 0
    moved_key = []
    moved_list = []
    for frame in manifest["frames"]:
        before = truth_for(frame)
        frame["faults_in_view"] = faults_in_view(defects, frame["theta_rad"], frame["z_m"])
        after = truth_for(frame)
        if before[0] != after[0]:
            moved_key.append((frame["station"], before, after))
        elif before != after:
            moved_list.append(frame["station"])
        for c in frame["probe_candidates"]:
            left, took = thickness_mm(defects, c["theta_rad"], c["z_m"])
            fraction = round(left / c["nominal_mm"], 4)
            was = (c["true_thickness_mm"], c["rejectable"])
            c["true_thickness_mm"] = left
            c["remaining_fraction"] = fraction
            c["thinned_by"] = took
            c["rejectable"] = fraction < limit
            if was != (left, c["rejectable"]):
                moved_points += 1
        frame["probe_on_seam"] = next(c for c in frame["probe_candidates"]
                                      if abs(c["offset_m"]) < 1e-9)

    now_rejectable = sum(1 for f in manifest["frames"]
                         for c in f["probe_candidates"] if c["rejectable"])
    now_thinned = sum(1 for f in manifest["frames"]
                      for c in f["probe_candidates"] if c["thinned_by"])
    total = sum(len(f["probe_candidates"]) for f in manifest["frames"])

    manifest["defects"] = defects
    manifest["answer_key"] = {
        "computed_by": "tank/truth.py and tank/thickness.py, from tank_scene.json",
        "frames_checked_against": DIGESTS.name,
        "seam_faults_projected_from": "fault_geometry.json, the rendered vertices",
        "projection_checked_against": "projection_check.json, Blender's own numbers",
    }
    manifest["probe_truth_refreshed_from"] = Path(args.scene).name
    manifest["evaluation_scene"] = (
        "The hidden wall loss on this run is heavier than a real shell of this "
        "age would carry. It is set that way so a probe planner can be measured "
        "against another one: with a single hidden patch the run held four "
        "rejectable points and no comparison on four points can separate "
        "anything.")

    print(f"{len(manifest['frames'])} frames, all matching {DIGESTS.name}")
    print(f"  stations whose worst-fault answer moved: {len(moved_key)}")
    for station, before, after in moved_key:
        print(f"    station {station}: {before[0]} -> {after[0]}")
    if moved_list:
        print(f"  stations where only the faults listed or grazed moved: "
              f"{', '.join(map(str, moved_list))}")
    print(f"{total} probe points")
    print(f"  thinned     {was_thinned:>4} -> {now_thinned}")
    print(f"  rejectable  {was_rejectable:>4} -> {now_rejectable}")
    print(f"  points whose answer moved: {moved_points}")

    if args.check:
        print("\n--check, nothing written")
        return 0
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {manifest_path}. Frames untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
