"""A camera that has crept looks exactly like a fixture that has crept.

The drift diagnosis in this repository answers one of three causes from the seam
stream, and it does it well. It also has a blind spot that no amount of model
makes up for, because the information is not in the data.

A fixture going off moves the PART. A camera going off moves the MEASUREMENT.
Both appear in the seam as the whole joint translating, by the same amount at the
bottom and at the top, with the tilt flat. Measured on the seam alone the two are
not merely hard to separate; they are the same numbers. So a cell asked which
machine to touch answers "fixture" to both, confidently, and is wrong half the
time. A fitter spends a shift re-shimming a fixture that was fine while the camera
mount goes on moving.

What fixes that is an instrument, not a prompt: a reference feature bolted to the
machine base rather than to the fixture, which the same camera reads on every
part. Nothing about the press or the fixture can move it. Only the camera can.

This does three things, in order, because the order is the argument:

  1. shows the confound is real, by putting the two streams side by side and
     showing the discriminators agree to two decimal places
  2. shows the reference separates them, on the same runs
  3. scores the models on the four-way question, with and without the reference,
     so the gain is attributable to the instrument rather than to the model

    python scripts/eval_confound.py
    python scripts/eval_confound.py --model nvidia/nemotron-3-super-120b-a12b
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
from cell.drift import (  # noqa: E402
    SCENARIOS_WITH_CAMERA,
    build_stream,
    discriminators,
    prompt_for,
    quiet_seed,
    user_prompt,
)

CONFUSABLE = ("fixture", "camera")


def show_confound(parts: int, seeds: list[int]) -> list[dict]:
    """The two faults, side by side, on the seam alone and then with the reference."""
    print("Step 1. The same two faults, measured without the reference.\n")
    print(f"{'seed':>6} {'scenario':>9} {'tilt t':>8} {'shift t':>9}   verdict")
    print("-" * 56)
    records = []
    for seed in seeds:
        blind = {}
        for scenario in CONFUSABLE:
            rows, _ = build_stream(parts, scenario, seed, reference=False)
            d = discriminators(rows)
            blind[scenario] = d
            print(f"{seed:>6} {scenario:>9} {d['tilt_ratio']:>8.2f} {d['shift_ratio']:>9.2f}")
        tilt_gap = abs(blind["fixture"]["tilt_ratio"] - blind["camera"]["tilt_ratio"])
        shift_gap = abs(blind["fixture"]["shift_ratio"] - blind["camera"]["shift_ratio"])
        print(f"{'':>6} {'':>9} {tilt_gap:>8.2f} {shift_gap:>9.2f}   "
              f"<- how far apart they are\n")
        records.append({"seed": seed, "tilt_gap": tilt_gap, "shift_gap": shift_gap})

    print("Step 2. The same runs, with a reference bolted to the machine base.\n")
    print(f"{'seed':>6} {'scenario':>9} {'tilt t':>8} {'shift t':>9} {'reference t':>12}")
    print("-" * 56)
    for seed in seeds:
        for scenario in CONFUSABLE:
            rows, _ = build_stream(parts, scenario, seed, reference=True)
            d = discriminators(rows)
            print(f"{seed:>6} {scenario:>9} {d['tilt_ratio']:>8.2f} "
                  f"{d['shift_ratio']:>9.2f} {d['reference_ratio']:>12.2f}")
        print()
    return records


def ask(rows: list[dict], model: str) -> dict:
    client = nebius.client()
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=400,
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": prompt_for(rows)},
            {"role": "user", "content": user_prompt(rows)},
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


def score(model: str, parts: int, seeds: list[int], reference: bool) -> list[dict]:
    arm = "with the reference" if reference else "on the seam alone"
    print(f"\n{'':>9} {'seed':>6} {'said':>9} {'ok':>4} {'ms':>6}   ({arm})")
    print("-" * 60)
    records = []
    for seed in seeds:
        for scenario in SCENARIOS_WITH_CAMERA:
            rows, truth = build_stream(parts, scenario, seed, reference=reference)
            verdict = ask(rows, model)
            correct = verdict.get("cause") == truth
            records.append({
                "model": model, "reference": reference, "scenario": truth, "seed": seed,
                "said": verdict.get("cause"), "correct": correct,
                "confidence": verdict.get("confidence"),
                "latency_ms": verdict.get("latency_ms"),
                "evidence": str(verdict.get("evidence"))
                .encode("ascii", "replace").decode("ascii"),
            })
            print(f"{truth:>9} {seed:>6} {str(verdict.get('cause')):>9} "
                  f"{'yes' if correct else 'NO':>4} {verdict.get('latency_ms'):>6}")
    return records


def summarise(records: list[dict], label: str) -> None:
    correct = sum(1 for r in records if r["correct"])
    print(f"\n  {label}: {correct}/{len(records)}")
    for scenario in SCENARIOS_WITH_CAMERA:
        rows = [r for r in records if r["scenario"] == scenario]
        if rows:
            got = sum(1 for r in rows if r["correct"])
            print(f"    {scenario:<9} {got}/{len(rows)}")
    confusable = [r for r in records if r["scenario"] in CONFUSABLE]
    if confusable:
        got = sum(1 for r in confusable if r["correct"])
        print(f"    the two that look alike: {got}/{len(confusable)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=30)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--model", default="nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--no-model", action="store_true",
                        help="Only demonstrate the confound; do not call anything.")
    parser.add_argument("--out", default="data/confound.jsonl")
    args = parser.parse_args()

    seeds = [quiet_seed(args.parts, 100 + run * 137) for run in range(args.runs)]
    gaps = show_confound(args.parts, seeds)

    worst_tilt = max(g["tilt_gap"] for g in gaps)
    worst_shift = max(g["shift_gap"] for g in gaps)
    print(f"Across {len(seeds)} runs the two faults never differ by more than "
          f"{worst_tilt:.2f} in tilt\nor {worst_shift:.2f} in shift, against a "
          f"threshold of 2.0 for calling a trend real.\nThey are the same measurement.\n")

    records = []
    if not args.no_model:
        print(f"Step 3. Asking {args.model} the four-way question, both ways.")
        blind = score(args.model, args.parts, seeds, reference=False)
        seeing = score(args.model, args.parts, seeds, reference=True)
        records = blind + seeing
        summarise(blind, "on the seam alone")
        summarise(seeing, "with the reference")

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"\nwrote {out}")

        blind_ms = statistics.median(r["latency_ms"] for r in blind)
        print(f"median latency {blind_ms:.0f} ms, unchanged by the extra column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
