"""Can the cell tell a drifting press from a drifting fixture, and from nothing?

Spotting that something is wrong is the easy half. The half that is worth money
is saying WHICH machine to go and touch, because the two faults that move a weld
seam are fixed at opposite ends of the shop and look similar in a summary number.

They do not look similar in the measurement stream, and that is the point:

  a fixture going off    moves the whole seam together. The bottom of the joint
                         and the top of it shift by the same amount, because the
                         part is sitting in the wrong place and is otherwise fine.

  a press going off      moves the top of the joint and leaves the bottom where
                         it was, because a bend angle is applied at a fold and
                         multiplied by height. The seam tilts rather than shifts.

So the discriminator is the difference between the top and the bottom of the
seam, which a cell that only locates a point cannot see and a cell that tracks the
whole seam gets for free. This scores whether a model reading that stream reaches
the right conclusion, against three scenarios where the answer is known: a press
drifting, a fixture drifting, and nothing wrong at all.

The control matters. A diagnostician that always finds something is not
diagnosing.

    python scripts/eval_drift.py --parts 30 --runs 3
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402
from brain.router import NANO  # noqa: E402
from cell import geometry as geo  # noqa: E402
from cell.forming import PressDrift, plan_shift  # noqa: E402

SYSTEM_PROMPT = """You watch a robot welding cell and diagnose what is going wrong.

Each part is measured before welding. The cell tracks the weld seam along its
whole length, so for every part you are given where the seam sits at the BOTTOM of
the joint and where it sits at the TOP, both as an offset in millimetres from
where a perfect part would put it.

Three things can be happening.

fixture: the part is being located badly. The whole seam moves together, so the
bottom offset and the top offset drift by about the same amount. Fix the fixture
or the locators.

press: the press brake is going out of adjustment. A bend angle is applied at the
fold and multiplied by the height of the side, so the TOP of the seam walks while
the BOTTOM stays roughly where it was. The tilt, top minus bottom, is what trends.
Fix the press.

none: nothing is trending. Parts scatter about zero the way any process does.
Say so. Do not invent a fault.

Answer with JSON only:
{"cause": "fixture" | "press" | "none",
 "confidence": 0.0 to 1.0,
 "evidence": "one or two sentences naming the numbers that decided it"}"""


def build_stream(parts: int, scenario: str, seed: int,
                 severity: float = 1.0) -> tuple[list[dict], str]:
    """Make a run of parts with a known fault, and the measurements it produces.

    severity scales how far out the machine has gone. At 1.0 the press drift
    moves the tilt by about twice the part-to-part scatter, which is a fault a
    shop would be arguing about rather than one it had already found.
    """
    rng = random.Random(seed)
    drift = (PressDrift(lean_deg_per_part=0.012 * severity, side="right",
                        starts_at=parts // 4)
             if scenario == "press" else None)
    trays = plan_shift(parts, seed=seed, drift=drift)

    rows = []
    for index, tray in enumerate(trays):
        corner = tray.welded_corner()

        # How the part sat in the fixture. Ordinary scatter, plus a creep in one
        # direction if the fixture is the thing going off.
        settle = rng.gauss(0.0, 0.004)
        if scenario == "fixture" and index >= parts // 4:
            settle += 0.0006 * severity * (index - parts // 4)

        # Offsets of the seam from nominal, at the bottom of the joint and the top,
        # measured along the direction the corner actually moves: outward on the
        # diagonal between the two sides.
        #
        # Taking only the y component, as this did at first, hides a fault on the
        # side that moves the seam in x, and the press scenario came out with
        # exactly the same numbers as the control. A fault the measurement cannot
        # see is not a fault the model can be scored on.
        root_two = math.sqrt(2.0)
        bottom_dx, bottom_dy = corner.seam_offset_m(0.0, geo.PART_D)
        top_dx, top_dy = corner.seam_offset_m(geo.PART_D, geo.PART_D)
        bottom = ((bottom_dx - bottom_dy) / root_two + settle) * 1000.0
        top = ((top_dx - top_dy) / root_two + settle) * 1000.0
        rows.append({
            "part": index,
            "seam_bottom_mm": round(bottom, 3),
            "seam_top_mm": round(top, 3),
            "tilt_mm": round(top - bottom, 3),
        })
    return rows, scenario


def ask(rows: list[dict], model: str) -> dict:
    client = nebius.client()
    table = "\n".join(
        f"part {r['part']:>3}  bottom {r['seam_bottom_mm']:+7.3f}  "
        f"top {r['seam_top_mm']:+7.3f}  tilt {r['tilt_mm']:+7.3f}"
        for r in rows
    )
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=400,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Measurements in the order the parts were made, millimetres:\n\n{table}"},
        ],
    )
    raw = nebius.reply_text(response.choices[0].message)
    latency = (time.perf_counter() - started) * 1000
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        data = {"cause": "unparsed", "confidence": 0.0, "evidence": raw[:160]}
    data["latency_ms"] = round(latency)
    return data


def trend(values: list[float]) -> float:
    """Least squares slope, as a sanity check the model has something to find."""
    n = len(values)
    xs = list(range(n))
    mx, my = statistics.mean(xs), statistics.mean(values)
    denominator = sum((x - mx) ** 2 for x in xs)
    return 0.0 if denominator == 0 else sum(
        (x - mx) * (y - my) for x, y in zip(xs, values, strict=True)) / denominator


def quiet_seed(parts: int, start: int) -> int:
    """Find a seed whose no-fault run really is flat.

    A control drawn at random is not automatically a control. Ordinary scatter
    produces trends by chance, and the first version of this scored the model
    against a run labelled "nothing wrong" that had a visible tilt in it. The
    model called that a press fault, which on the evidence in front of it was a
    reasonable answer to a question that had been asked wrongly.

    So the seed is chosen by checking that the run it produces has no trend worth
    finding, before anything is asked about it.
    """
    for offset in range(400):
        seed = start + offset
        rows, _ = build_stream(parts, "none", seed)
        if (abs(trend([r["tilt_mm"] for r in rows])) < 0.008
                and abs(trend([r["seam_bottom_mm"] for r in rows])) < 0.030):
            return seed
    raise SystemExit("no quiet seed found; widen the search")


def safe(text: object) -> str:
    """Model replies carry characters a Windows console cannot print."""
    return str(text).encode("ascii", "replace").decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default=NANO)
    parser.add_argument("--severity", type=float, default=1.0,
                        help="How far out the machine has gone. 1.0 is marginal.")
    parser.add_argument("--out", default="data/drift_eval.jsonl")
    args = parser.parse_args()

    scenarios = ("press", "fixture", "none")
    records = []
    print(f"{'scenario':>9} {'seed':>5} {'tilt slope':>11} {'shift slope':>12} "
          f"{'said':>9} {'conf':>6} {'ok':>4}  evidence")
    print("-" * 118)

    for run in range(args.runs):
        for scenario in scenarios:
            seed = quiet_seed(args.parts, 100 + run * 137)
            rows, truth = build_stream(args.parts, scenario, seed, args.severity)
            verdict = ask(rows, args.model)
            correct = verdict.get("cause") == truth
            tilt_slope = trend([r["tilt_mm"] for r in rows])
            shift_slope = trend([r["seam_bottom_mm"] for r in rows])
            records.append({
                "scenario": truth, "seed": seed, "said": verdict.get("cause"),
                "confidence": verdict.get("confidence"), "correct": correct,
                "tilt_slope_mm_per_part": round(tilt_slope, 4),
                "shift_slope_mm_per_part": round(shift_slope, 4),
                "evidence": safe(verdict.get("evidence")),
                "latency_ms": verdict.get("latency_ms"),
            })
            print(f"{truth:>9} {seed:>5} {tilt_slope:>10.4f} {shift_slope:>11.4f} "
                  f"{str(verdict.get('cause')):>9} {str(verdict.get('confidence')):>6} "
                  f"{'yes' if correct else 'NO':>4}  {safe(verdict.get('evidence'))[:52]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    correct = sum(1 for r in records if r["correct"])
    print(f"\n{correct}/{len(records)} diagnosed correctly")
    for scenario in scenarios:
        rows = [r for r in records if r["scenario"] == scenario]
        got = sum(1 for r in rows if r["correct"])
        print(f"  {scenario:<8} {got}/{len(rows)}")
    print(f"  median latency {statistics.median(r['latency_ms'] for r in records):.0f} ms")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
