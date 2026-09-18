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
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402
from brain.router import NANO  # noqa: E402
from cell.drift import (  # noqa: E402
    SCENARIOS,
    SYSTEM_PROMPT,
    build_stream,
    parse_verdict,
    quiet_seed,
    trend,
    user_prompt,
)


def ask(rows: list[dict], model: str) -> dict:
    client = nebius.client()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=400,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(rows)},
        ],
    )
    raw = nebius.reply_text(response.choices[0].message)
    latency = (time.perf_counter() - started) * 1000
    data = parse_verdict(raw, SCENARIOS)
    data["latency_ms"] = round(latency)
    return data


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

    scenarios = SCENARIOS
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
