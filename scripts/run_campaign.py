"""Run a measurement campaign over a captured manifest, and score it.

Two arms per part, on identical data:

  baseline  one fixed view, which is what a cell with a bolted-down camera gets
  guided    look, ask the cloud whether to believe it, look again if not, fuse

The comparison is honest because both arms see the same renders of the same parts,
and because turning the part about the joint axis leaves the joint where it is, so
the target never moves between views.

    python scripts/run_campaign.py --manifest data/captures/manifest.json

Add --no-post to skip the service. The planner needs a Token Factory key.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.client import CellClient  # noqa: E402
from cell.depth_joint import (  # noqa: E402
    Intrinsics,
    measure_joint,
    project,
    working_volume,
)

BASELINE_VIEW_DEG = 0.0
JOINT_DIRECTION = np.array([0.0, 0.0, 1.0])
ROI_HALF_WIDTH_PX = 120
ROI_HALF_HEIGHT_PX = 140
RANGE_MARGIN_M = 0.25


@dataclass
class Context:
    root: Path
    intrinsics: Intrinsics
    to_world: np.ndarray
    joint_height: float
    roi_centre: np.ndarray
    expected_range: float


@dataclass
class Look:
    """One attempt at measuring, successful or not."""

    view_deg: float
    joint_mm: Optional[list[float]] = None
    signals: dict[str, Any] = field(default_factory=dict)
    error_mm: Optional[float] = None
    failed: Optional[str] = None

    def for_planner(self) -> dict[str, Any]:
        """What the model is shown. Never the error: that is ground truth."""
        entry: dict[str, Any] = {"view_deg": self.view_deg}
        if self.failed:
            entry["measurement_failed"] = self.failed
        else:
            entry["signals"] = self.signals
            entry["joint_mm"] = [round(v, 3) for v in self.joint_mm]
        return entry


def load_context(manifest_path: Path) -> tuple[dict, Context]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("rendered"):
        raise SystemExit("this manifest has no images; run the station without --no-render")
    camera = manifest["camera"]

    intrinsics = Intrinsics.from_camera(
        camera["width"],
        camera["height"],
        camera["focal_length_mm"],
        camera["horizontal_aperture_mm"],
        camera["vertical_aperture_mm"],
    )
    to_world = np.array(camera["to_world"])
    height = manifest["joint_height"]

    # Where the taught path says the joint is. A position prior only.
    nominal = geo.nominal_joint()
    centre, distance = project(
        np.array([[nominal[0], nominal[1], height]]), intrinsics, to_world
    )
    return manifest, Context(
        root=manifest_path.parent,
        intrinsics=intrinsics,
        to_world=to_world,
        joint_height=height,
        roi_centre=centre[0],
        expected_range=float(distance[0]),
    )


def take_look(view: dict, truth: np.ndarray, ctx: Context) -> Look:
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
    try:
        measurement = measure_joint(
            depth,
            ctx.intrinsics,
            ctx.to_world,
            mask=mask,
            expected_direction=JOINT_DIRECTION,
            rng=np.random.default_rng(0),
        )
    except RuntimeError as exc:
        return Look(view_deg=view["orbit_deg"], failed=str(exc))

    joint = np.array(measurement.at_height(ctx.joint_height))
    return Look(
        view_deg=view["orbit_deg"],
        joint_mm=[joint[0] * 1000.0, joint[1] * 1000.0],
        signals=measurement.summary(),
        error_mm=1000.0 * float(np.linalg.norm(joint - truth)),
    )


def fuse(looks: list[Look]) -> Optional[np.ndarray]:
    """Median of the accepted estimates, per axis.

    A median rather than a mean, because the failure that matters is one view
    landing far away while the rest agree, and a mean would split the difference.
    """
    points = [look.joint_mm for look in looks if look.joint_mm]
    if not points:
        return None
    return np.array(
        [
            statistics.median(p[0] for p in points),
            statistics.median(p[1] for p in points),
        ]
    ) / 1000.0


def run_guided(
    row: dict, ctx: Context, max_looks: int, start_view_deg: float = BASELINE_VIEW_DEG
) -> tuple[list[Look], list[dict]]:
    from brain.viewplanner import plan_next_view

    truth = np.array(row["true_joint"])
    by_angle = {v["orbit_deg"]: v for v in row["views"]}
    available = sorted(by_angle)

    current = start_view_deg if start_view_deg in by_angle else available[0]
    looks: list[Look] = []
    decisions: list[dict] = []

    for _ in range(max_looks):
        looks.append(take_look(by_angle[current], truth, ctx))
        remaining = [a for a in available if a not in {look.view_deg for look in looks}]
        if not remaining:
            break

        decision = plan_next_view([look.for_planner() for look in looks], remaining)
        decisions.append(
            {
                "after_looks": len(looks),
                "accept": decision.accept,
                "next_view_deg": decision.next_view_deg,
                "reason": decision.reason,
                "latency_ms": round(decision.latency_ms or 0.0),
            }
        )
        if decision.accept or decision.next_view_deg is None:
            break
        current = decision.next_view_deg

    return looks, decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/captures/manifest.json")
    parser.add_argument("--max-looks", type=int, default=4)
    parser.add_argument(
        "--baseline-view",
        type=float,
        default=BASELINE_VIEW_DEG,
        help="Which fixed view the baseline arm gets, and where the guided arm "
        "starts. A cell with a bolted-down camera does not get to choose.",
    )
    parser.add_argument("--out", default="data/campaign.jsonl")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--api", default=None)
    args = parser.parse_args()

    manifest, ctx = load_context(Path(args.manifest))
    client = None if args.no_post else CellClient(base_url=args.api)

    results = []
    started = time.perf_counter()

    print(f"{'part':>5} {'baseline':>10} {'guided':>10} {'looks':>6} {'spread':>8}  decision")
    print("-" * 96)

    for row in manifest["rows"]:
        truth = np.array(row["true_joint"])
        by_angle = {v["orbit_deg"]: v for v in row["views"]}

        baseline_view = by_angle.get(args.baseline_view) or row["views"][0]
        baseline = take_look(baseline_view, truth, ctx)

        looks, decisions = run_guided(row, ctx, args.max_looks, args.baseline_view)
        fused = fuse(looks)
        guided_error = None if fused is None else 1000.0 * float(np.linalg.norm(fused - truth))

        used = [look for look in looks if look.joint_mm]
        spread = None
        if len(used) > 1:
            xs = [look.joint_mm[0] for look in used]
            ys = [look.joint_mm[1] for look in used]
            spread = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))

        record = {
            "trial": row["trial"],
            "true_joint": row["true_joint"],
            "blind_error_mm": row["blind_error_mm"],
            "baseline_view_deg": baseline.view_deg,
            "baseline_error_mm": baseline.error_mm,
            "baseline_failed": baseline.failed,
            "guided_error_mm": guided_error,
            "looks": [
                {
                    "view_deg": look.view_deg,
                    "error_mm": look.error_mm,
                    "failed": look.failed,
                    "signals": look.signals,
                }
                for look in looks
            ],
            "decisions": decisions,
            "spread_mm": spread,
        }
        results.append(record)

        note = decisions[-1]["reason"][:52] if decisions else "single view only"
        print(
            f"{row['trial']:5d} "
            f"{'fail' if baseline.error_mm is None else f'{baseline.error_mm:9.3f}':>10} "
            f"{'fail' if guided_error is None else f'{guided_error:9.3f}':>10} "
            f"{len(looks):6d} "
            f"{'-' if spread is None else f'{spread:7.3f}':>8}  {note}"
        )

        if client is not None and fused is not None:
            serial = client.create_part(
                station="sim",
                spawn_dy_mm=row["spawn_dy_mm"],
                spawn_yaw_deg=row["spawn_yaw_deg"],
            )
            if serial:
                client.post_measurement(
                    serial,
                    detected=(float(fused[0]), float(fused[1])),
                    confidence=1.0,
                    truth=(truth[0], truth[1]),
                    n_scans=len(used),
                    spread_mm=spread,
                )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record) + "\n")

    summarise(results)
    print(f"\nwrote {args.out} in {time.perf_counter() - started:.1f} s")
    return 0


def summarise(results: list[dict]) -> None:
    def stats(values: list[float]) -> str:
        if not values:
            return "no successful measurements"
        ordered = sorted(values)
        p95 = ordered[max(0, min(len(ordered) - 1, round(0.95 * len(ordered) + 0.5) - 1))]
        return (
            f"n={len(values):3d}  mean {statistics.fmean(values):7.3f}  "
            f"median {statistics.median(values):7.3f}  p95 {p95:7.3f}  "
            f"max {max(values):7.3f}  sub-mm {sum(v < 1.0 for v in values)}/{len(values)}"
        )

    baseline = [r["baseline_error_mm"] for r in results if r["baseline_error_mm"] is not None]
    guided = [r["guided_error_mm"] for r in results if r["guided_error_mm"] is not None]
    blind = [r["blind_error_mm"] for r in results]
    looks = [len(r["looks"]) for r in results]

    print("\nerror against the true joint, in millimetres")
    print(f"  no correction   {stats(blind)}")
    print(f"  baseline        {stats(baseline)}")
    print(f"  cloud guided    {stats(guided)}")
    print(f"\n  baseline failed to measure on {len(results) - len(baseline)} of {len(results)}")
    print(f"  guided failed to measure on   {len(results) - len(guided)} of {len(results)}")
    if looks:
        print(f"  looks per part: mean {statistics.fmean(looks):.2f}, max {max(looks)}")


if __name__ == "__main__":
    raise SystemExit(main())
