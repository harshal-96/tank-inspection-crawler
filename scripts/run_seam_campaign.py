"""Track the seam along its whole length, on every captured part, and score it.

The joint campaign answers a narrower question: where is the corner. This one
answers the question the cell exists for: where does the seam RUN. A part whose
corner is located perfectly can still be welded badly, because springback and
camber bow the seam away from the straight line between its own ends, and a path
that is straight by construction cannot follow that.

Two arms on identical data:

  taught    the straight path a fixed program drives, between the seam ends of a
            nominal part sitting where the fixture says it will
  tracked   the path measured from depth, station by station up the seam

Scored against the true seam, which is known only because the press error that
produced it was chosen by the simulation.

    python scripts/run_seam_campaign.py --manifest data/cap50v3/manifest.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.capture import load_depth  # noqa: E402
from cell.client import CellClient  # noqa: E402
from cell.depth_joint import Intrinsics, backproject, transform_points  # noqa: E402
from cell.seam_track import find_and_track  # noqa: E402
from cell.thermal import ThermalDistortion, WeldPass  # noqa: E402
from cell.weld_params import (  # noqa: E402
    ProcessLimits,
    excess_gap_mm,
    schedule_along_seam,
)


def straight_line_miss(truth: np.ndarray) -> float:
    """How far the true seam strays from the straight line between its own ends.

    This is the part of the error that no amount of touching up a single point
    can fix. A seam that is merely shifted can be corrected by locating the part;
    a seam that wanders has to be followed.
    """
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
    """Distance from each measured station to the true seam at its own height."""
    errors = []
    for station in stations:
        point = station.position
        nearest = truth[int(np.argmin(np.abs(truth[:, 2] - point[2])))]
        errors.append(float(np.linalg.norm(nearest[:2] - point[:2]) * 1000.0))
    return {
        "mean_mm": statistics.mean(errors),
        "max_mm": max(errors),
        "errors_mm": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cap50v3/manifest.json")
    parser.add_argument("--view", type=float, default=0.0,
                        help="Which candidate view to track from.")
    parser.add_argument("--stations", type=int, default=20)
    parser.add_argument("--out", default="data/seam_campaign.jsonl")
    parser.add_argument("--no-post", action="store_true",
                        help="Do not send results to the service.")
    parser.add_argument("--api", default=None, help="Service base URL")
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

    # The taught path: straight, vertical, and where the fixture says the part
    # will be. It starts clear of the base, which occludes the bottom of the
    # corner from any camera standing outside the part.
    nominal = geo.nominal_joint()
    floor = geo.TABLE_TOP + geo.PART_T
    taught_start = np.array([nominal[0], nominal[1], floor + 0.04])
    taught_end = np.array([nominal[0], nominal[1], floor + geo.PART_D - 0.02])

    client = None
    if not args.no_post:
        client = CellClient(base_url=args.api)
        if not client.healthy():
            print("[seam] service not reachable, results will be queued to disk")

    limits = ProcessLimits(thickness_mm=geo.PART_T * 1000.0)

    started = time.time()
    records = []
    print(f"{'part':>5} {'taught':>9} {'tracked mean':>13} {'worst':>8} "
          f"{'stations':>9}  {'seam bow':>9}")
    print("-" * 64)

    for row in manifest["rows"]:
        view = min(row["views"], key=lambda v: abs(v["orbit_deg"] - args.view))
        depth = load_depth(root / view["depth"])
        # Timed from the depth frame to a finished path, which is the work the cell
        # does between the camera firing and the torch being allowed to move. The
        # file read is outside it; on a real sensor that is a frame arriving.
        track_from = time.perf_counter()
        cloud = transform_points(backproject(depth, intrinsics), to_world)
        track = find_and_track(cloud, taught_start, taught_end, stations=args.stations)
        track_ms = (time.perf_counter() - track_from) * 1000.0

        truth = np.array(row["true_seam"])
        # Score over the stretch the camera can actually see.
        truth = truth[truth[:, 2] >= taught_start[2]]

        record = {
            "trial": row["trial"],
            "taught_miss_mm": row["taught_miss_mm"],
            "seam_bow_mm": straight_line_miss(truth),
            "stations_measured": len(track.measured),
            "stations_asked": args.stations,
            "track_ms": round(track_ms, 1),
        }
        if track.measured:
            record.update(score_path(track.measured, truth))
            record["measured_wander_mm"] = float(track.wander_mm())
        # What the torch should do about the gap it just measured, station by
        # station, because the gap is not the same all the way up.
        # Referenced against what a touching joint of this geometry reads, which
        # is a sensor calibration rather than a fudge. See cell/weld_params.py for
        # the measurement it comes from and the uncertainty it leaves.
        gaps = [excess_gap_mm(s.gap_mm) for s in track.measured]
        schedules = schedule_along_seam(gaps, limits) if gaps else []
        record["refused_stations"] = sum(1 for s in schedules if not s.weldable)

        if client is not None and track.measured:
            serial = f"P-{row['trial']:06d}"
            client.create_part(
                serial=serial,
                spawn_dy_mm=row.get("spawn_dy_mm"),
                spawn_yaw_deg=row.get("spawn_yaw_deg"),
            )
            client.post_seam_track(
                part_serial=serial,
                path=[
                    {
                        "along_mm": station.along_m * 1000.0,
                        "x_mm": float(station.position[0]) * 1000.0,
                        "y_mm": float(station.position[1]) * 1000.0,
                        "gap_mm": station.gap_mm,
                        "touching": station.touching,
                    }
                    for station in track.measured
                ],
                stations_asked=args.stations,
                wander_mm=record.get("measured_wander_mm"),
                taught_miss_mm=row["taught_miss_mm"],
                gap_mm_mean=statistics.mean(gaps) if gaps else None,
                gap_mm_max=max(gaps) if gaps else None,
                seam_bow_mm=record["seam_bow_mm"],
                error_mm=record.get("mean_mm"),
                worst_station_mm=record.get("max_mm"),
                latency_ms=track_ms,
            )
            # A seam can come back with every station refused, and asking for the
            # hottest of nothing raises rather than reporting the refusal.
            weldable = [s for s in schedules if s.weldable]
            if weldable:
                hottest = max(s.heat_input_j_per_mm for s in weldable)
                slowest = min(s.travel_mm_s for s in weldable)
                drift = ThermalDistortion(
                    weld=WeldPass(travel_mm_s=slowest),
                    thickness_mm=limits.thickness_mm,
                    seam_length_mm=geo.PART_D * 1000.0,
                ).drift_at_finish_mm()
                client.post_weld_schedule(
                    part_serial=serial,
                    stations=[
                        {
                            "along_mm": station.along_m * 1000.0,
                            "gap_mm": plan.gap_mm,
                            "current_a": plan.current_a,
                            "voltage_v": plan.voltage_v,
                            "travel_mm_s": plan.travel_mm_s,
                            "heat_j_per_mm": plan.heat_input_j_per_mm,
                            "weldable": plan.weldable,
                            "needs_attention": plan.needs_attention,
                            "reason": plan.reason,
                        }
                        for station, plan in zip(track.measured, schedules, strict=True)
                    ],
                    predicted_drift_mm=drift,
                )
                record["hottest_j_per_mm"] = hottest

        records.append(record)

        print(f"{row['trial']:>5} {row['taught_miss_mm']:>7.1f} mm "
              f"{record.get('mean_mm', float('nan')):>11.2f} mm "
              f"{record.get('max_mm', float('nan')):>6.2f} mm "
              f"{len(track.measured):>5}/{args.stations:<3} "
              f"{record['seam_bow_mm']:>7.2f} mm")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    scored = [r for r in records if "mean_mm" in r]
    taught = [r["taught_miss_mm"] for r in records]
    tracked = [r["mean_mm"] for r in scored]
    worst = [r["max_mm"] for r in scored]
    bow = [r["seam_bow_mm"] for r in records]

    def line(name, values):
        ordered = sorted(values)
        p95 = ordered[max(0, int(round(0.95 * len(ordered))) - 1)]
        print(f"  {name:<16} n={len(ordered):>3}  mean {statistics.mean(ordered):7.3f}  "
              f"median {statistics.median(ordered):7.3f}  p95 {p95:7.3f}  "
              f"max {ordered[-1]:7.3f}  sub-mm {sum(1 for v in ordered if v < 1.0)}/{len(ordered)}")

    print("\ndistance from the welded path to the true seam, in millimetres")
    line("taught path", taught)
    line("tracked path", tracked)
    print(f"\n  worst station on any part: {max(worst):.2f} mm")
    print(f"  the seam itself bows off its own straight line by "
          f"{statistics.mean(bow):.2f} mm on average, worst {max(bow):.2f} mm")
    print(f"  parts fully tracked: "
          f"{sum(1 for r in records if r['stations_measured'] == args.stations)}/{len(records)}")
    print(f"\nwrote {out} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
