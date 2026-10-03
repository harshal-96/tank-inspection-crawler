"""Where does this method stop working?

Every number in this repository so far was measured on exact rendered depth. That
makes them ceilings rather than results: a joint finder scored on perfect input
has been scored on nothing, and the honest question about a perception method is
not how well it does at its best but where the edge of it is.

So the whole fifty-part campaign is run again at each of a series of sensor
degradations, and the error curve is followed until the method fails. Failure is
defined before the sweep is run, in two ways, because they are different problems:

  drift     the tracked path is still produced but is no longer accurate enough
            to weld on. The threshold is one millimetre, which is the wire
            diameter: past that the arc is no longer on the joint.

  collapse  stations stop being measurable at all. This is the safer failure and
            the cell already handles it, because a station it cannot measure is a
            station it refuses rather than guesses at.

A method that drifts silently is dangerous. A method that collapses loudly is
merely stopped. Which of the two happens first, as the sensor gets worse, is the
thing worth knowing, and it is not obvious in advance.

    python scripts/eval_robustness.py
    python scripts/eval_robustness.py --axis dropout --limit 20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.capture import load_depth  # noqa: E402
from cell.depth_joint import Intrinsics, backproject, transform_points  # noqa: E402
from cell.seam_track import find_and_track  # noqa: E402
from cell.sensor import CLEAN, HARSH_SPECULAR, TYPICAL_STEREO, DepthSensor  # noqa: E402

# The wire is 1.0 mm across, so a path further than that from the joint is not on
# the joint any more. Decided here, before the sweep runs, rather than read off
# whichever curve came back.
WELDABLE_MM = 1.0

# Each axis is swept with everything else clean, so the curve shows what that one
# failure mode costs rather than a mixture.
AXES: dict[str, tuple[str, str, list[float]]] = {
    "axial": ("axial_mm_at_1m", "range noise at 1 m, mm",
              [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]),
    "dropout": ("dropout", "fraction of the frame lost to specular dropout",
                [0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90]),
    "flying": ("flying_pixels", "fraction of edge pixels reporting an intermediate range",
               [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]),
}

NAMED = {"clean": CLEAN, "typical_stereo": TYPICAL_STEREO, "harsh_specular": HARSH_SPECULAR}


def straight_line_miss(truth: np.ndarray) -> float:
    first, last = truth[0], truth[-1]
    axis = last - first
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return 0.0
    unit = axis / length
    rel = truth - first
    along = rel @ unit
    off = rel - np.outer(along, unit)
    return float(np.linalg.norm(off, axis=1).max() * 1000.0)


def score_path(stations, truth: np.ndarray) -> dict:
    errors = []
    for station in stations:
        point = station.position
        nearest = truth[int(np.argmin(np.abs(truth[:, 2] - point[2])))]
        errors.append(float(np.linalg.norm(nearest[:2] - point[:2]) * 1000.0))
    return {"mean_mm": statistics.mean(errors), "max_mm": max(errors)}


def run_campaign(rows, root, intrinsics, to_world, taught_start, taught_end,
                 sensor: DepthSensor, stations: int, seed: int,
                 refine: bool = True) -> dict:
    """The whole campaign at one sensor setting."""
    rng = np.random.default_rng(seed)
    means, worsts, measured, asked, tracked_parts, lost_parts = [], [], 0, 0, 0, 0

    for row in rows:
        view = min(row["views"], key=lambda v: abs(v["orbit_deg"]))
        depth = load_depth(root / view["depth"])
        frame = sensor.degrade(depth, rng)
        cloud = transform_points(backproject(frame, intrinsics), to_world)

        truth = np.array(row["true_seam"])
        truth = truth[truth[:, 2] >= taught_start[2]]

        asked += stations
        try:
            track = find_and_track(cloud, taught_start, taught_end,
                                   stations=stations, refine=refine)
        except Exception:  # noqa: BLE001 - a frame too poor to fit is a result
            lost_parts += 1
            continue
        if not track.measured:
            lost_parts += 1
            continue

        scored = score_path(track.measured, truth)
        means.append(scored["mean_mm"])
        worsts.append(scored["max_mm"])
        measured += len(track.measured)
        tracked_parts += 1

    if not means:
        return {"parts_tracked": 0, "parts_lost": lost_parts, "stations_measured": 0,
                "stations_asked": asked, "mean_mm": None, "worst_mm": None,
                "sub_mm_parts": 0, "weldable": False, "collapsed": True}

    mean = statistics.fmean(means)
    return {
        "parts_tracked": tracked_parts,
        "parts_lost": lost_parts,
        "stations_measured": measured,
        "stations_asked": asked,
        "station_yield": measured / asked if asked else 0.0,
        "mean_mm": mean,
        "median_mm": statistics.median(means),
        "worst_mm": max(worsts),
        "sub_mm_parts": sum(1 for w in worsts if w < WELDABLE_MM),
        "weldable": mean < WELDABLE_MM,
        "collapsed": tracked_parts < len(rows) * 0.9,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cap50v3/manifest.json")
    parser.add_argument("--stations", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N parts")
    parser.add_argument("--axis", choices=[*AXES, "all", "named"], default="all")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", default="data/robustness.jsonl")
    parser.add_argument("--no-refine", action="store_true",
                        help="Score the slice measurements alone, without fitting the "
                             "faces from all their points. This is the arm the "
                             "improvement is measured against.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    camera = manifest["camera"]
    intrinsics = Intrinsics.from_camera(
        camera["width"], camera["height"], camera["focal_length_mm"],
        camera["horizontal_aperture_mm"], camera["vertical_aperture_mm"],
    )
    to_world = np.array(camera["to_world"])

    nominal = geo.nominal_joint()
    floor = geo.TABLE_TOP + geo.PART_T
    taught_start = np.array([nominal[0], nominal[1], floor + 0.04])
    taught_end = np.array([nominal[0], nominal[1], floor + geo.PART_D - 0.02])

    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]
    arm = "slices alone" if args.no_refine else "faces fitted from all their points"
    print(f"{len(rows)} parts, {args.stations} stations each, {arm}.")
    print(f"Weldable means mean error under {WELDABLE_MM:.1f} mm, the wire diameter.\n")

    started = time.time()
    records = []

    def report(axis: str, label: str, value, sensor: DepthSensor) -> None:
        result = run_campaign(rows, root, intrinsics, to_world, taught_start,
                              taught_end, sensor, args.stations, args.seed,
                              refine=not args.no_refine)
        result.update({"axis": axis, "label": label, "value": value,
                       "refined": not args.no_refine, "sensor": sensor.__dict__})
        records.append(result)
        mean = "lost" if result["mean_mm"] is None else f"{result['mean_mm']:7.3f}"
        worst = "    -" if result["worst_mm"] is None else f"{result['worst_mm']:6.2f}"
        verdict = ("COLLAPSED" if result["parts_tracked"] == 0
                   else "weldable" if result["weldable"] else "DRIFTED")
        print(f"  {label:>28}  mean {mean} mm  worst {worst} mm  "
              f"parts {result['parts_tracked']:>3}/{len(rows)}  "
              f"yield {result.get('station_yield', 0):5.1%}  {verdict}")

    axes = list(AXES) if args.axis in ("all", "named") else [args.axis]
    if args.axis != "named":
        for axis in axes:
            field, description, values = AXES[axis]
            print(f"[{axis}] {description}")
            for value in values:
                report(axis, f"{field}={value:g}", value, replace(CLEAN, **{field: value}))
            print()

    if args.axis in ("all", "named"):
        print("[named] whole sensors, every failure mode at once")
        for name, sensor in NAMED.items():
            report("named", name, name, sensor)
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    for axis in axes:
        curve = [r for r in records if r["axis"] == axis]
        if not curve:
            continue
        held = [r for r in curve if r["weldable"]]
        broke = [r for r in curve if not r["weldable"]]
        limit = max((r["value"] for r in held), default=None)
        first_fail = min((r["value"] for r in broke), default=None)
        how = ("nothing measurable" if broke and broke[0]["parts_tracked"] == 0
               else "accuracy, while still reporting a path")
        print(f"{axis}: holds to {limit}, fails at {first_fail} by {how}"
              if first_fail is not None else
              f"{axis}: weldable across the whole sweep, up to {limit}")

    print(f"\nwrote {out} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
