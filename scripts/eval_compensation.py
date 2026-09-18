"""Does welding where the joint is GOING to be beat welding where it was?

The cell measures cold and welds hot. Heat put into the bottom of the seam rotates
both folded sides about their fold lines, so the top of the joint walks inward
before the torch reaches it. The thermal model predicts 1.84 mm on this part, and
until now nothing used the prediction: the path was driven to the cold
measurement, which is where the joint used to be.

Compensating is easy. Proving it helps is the part worth doing carefully, because
there is an obvious way to cheat: predict the distortion with the same model that
produces it, and the correction is exact by construction. That measures nothing.

So the two models are deliberately different. The metal moves by constants drawn
off nominal by a stated amount; the cell compensates using the nominal ones,
because that is all a real cell has. The question then is a real one, with two
sides to it:

  under-modelled    the correction is too small and some of the walk survives
  over-modelled     the correction overshoots and pushes the torch past the joint

Both are failures and both are reported. The output is the range of model error
over which compensating still beats not compensating, which is the number that
says whether this is worth doing on a real machine.

    python scripts/eval_compensation.py
    python scripts/eval_compensation.py --trials 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.compensate import compensated_path, where_the_seam_ends_up  # noqa: E402
from cell.thermal import ThermalDistortion, WeldPass  # noqa: E402

# How far the truth's constants are drawn from nominal. A calibrated distortion
# model is not accurate to a few percent, so these are wide on purpose.
MODEL_ERROR_STEPS = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

WELDABLE_MM = 1.0


def seam_error_mm(path: np.ndarray, target: np.ndarray) -> list[float]:
    """Distance from each welded point to where the joint actually ended up."""
    out = []
    for point in path:
        nearest = target[int(np.argmin(np.abs(target[:, 2] - point[2])))]
        out.append(float(np.linalg.norm(nearest[:2] - point[:2]) * 1000.0))
    return out


def build_truth(stations: int) -> tuple[np.ndarray, np.ndarray]:
    """A straight cold seam and the station heights along it.

    The cold seam is taken as straight here on purpose. This study is about one
    thing, the walk the arc puts into the joint, and giving the seam a bow as well
    would mix in the error the seam tracker already handles and is scored on
    elsewhere.
    """
    floor = geo.TABLE_TOP + geo.PART_T
    heights = np.linspace(0.02, geo.PART_D - 0.02, stations)
    corner = geo.nominal_joint()
    cold = np.column_stack([
        np.full(stations, corner[0]),
        np.full(stations, corner[1]),
        floor + heights,
    ])
    # Measured from the strike, which is the first station, because that is where
    # the thermal model starts counting heat.
    return cold, (cold[:, 2] - cold[:, 2].min()) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", default="data/compensation.jsonl")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cold, along_mm = build_truth(args.stations)
    floor = cold[:, 2].min()

    # What the cell believes, and welds with. One model, fixed, nominal.
    believed = ThermalDistortion(
        weld=WeldPass(), thickness_mm=geo.PART_T * 1000.0,
        seam_length_mm=float(along_mm[-1]),
    )
    print(f"the cell predicts the corner walks {believed.drift_at_finish_mm():.2f} mm "
          f"by the end of the weld\n")
    print(f"{'model error':>12} {'uncompensated':>14} {'compensated':>13} "
          f"{'change':>9} {'worst':>8}  verdict")
    print("-" * 74)

    records = []
    for step in MODEL_ERROR_STEPS:
        raw, fixed, worst_fixed = [], [], []
        for _ in range(args.trials):
            # The metal's own constants: off nominal by the step, plus scatter,
            # because no two parts distort identically either.
            scale = (1.0 + step) * float(rng.normal(1.0, 0.08))
            actual = replace(
                believed,
                weld=replace(believed.weld, current_a=believed.weld.current_a * scale),
            )
            target = where_the_seam_ends_up(cold, actual, seam_floor_m=floor)

            raw.append(statistics.fmean(seam_error_mm(cold, target)))
            moved, _ = compensated_path(cold, along_mm, believed)
            errors = seam_error_mm(moved, target)
            fixed.append(statistics.fmean(errors))
            worst_fixed.append(max(errors))

        before, after = statistics.fmean(raw), statistics.fmean(fixed)
        worst = statistics.fmean(worst_fixed)
        helped = after < before
        verdict = ("helps" if helped and after < WELDABLE_MM
                   else "helps, still off" if helped
                   else "OVERSHOOTS")
        records.append({
            "model_error": step, "uncompensated_mm": before, "compensated_mm": after,
            "worst_compensated_mm": worst, "helped": helped,
            "weldable": after < WELDABLE_MM, "trials": args.trials,
        })
        print(f"{step:>+11.0%} {before:>13.3f} mm {after:>10.3f} mm "
              f"{after - before:>+8.3f} {worst:>7.2f}  {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    helped = [r["model_error"] for r in records if r["helped"]]
    weldable = [r["model_error"] for r in records if r["weldable"]]
    nominal = next(r for r in records if r["model_error"] == 0.0)
    print(f"\nat nominal the correction takes the miss from "
          f"{nominal['uncompensated_mm']:.2f} mm to {nominal['compensated_mm']:.2f} mm")
    if helped:
        print(f"compensating still beats not compensating from "
              f"{min(helped):+.0%} to {max(helped):+.0%} model error")
    if weldable:
        print(f"and stays inside the {WELDABLE_MM:.1f} mm wire from "
              f"{min(weldable):+.0%} to {max(weldable):+.0%}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
