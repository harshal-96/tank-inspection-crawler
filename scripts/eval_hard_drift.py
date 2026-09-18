"""The drift question on cases that sit ON the decision boundary.

Why this exists. The published evaluation has no marginal case in it: every run
sits at least 0.58 away from the threshold and most sit one to two away. A fixed
rule cannot fail on it, so every score measured against it is a score on a test a
ruler passes, and the differences between models were differences in how well each
guessed a boundary the prompt never stated. Telling a frontier model the boundary
took it from 9 of 16 to 16 of 16, and four lines of arithmetic also scored 16 of
16, instantly. That is not a benchmark, it is a tautology.

So this builds cases where a ruler and a diagnostician come apart:

  clear          a fault well clear of the threshold, kept as a control
  marginal       a real fault that does NOT clear the threshold
  lucky_noise    nothing injected, but scatter pushes a channel over the line
  subtle_camera  a camera creeping near the line rather than far past it
  mixed          a press AND a fixture going out together

Four arms are scored on the same cases:

  rule           the decision boundary in four lines, no model
  cloud          a Nemotron, told the boundary explicitly
  cloud_blind    the same Nemotron, boundary withheld, as the studies had it
  edge           the fine-tuned model, if its adapter is present

On the mixed family, naming EITHER genuine fault counts as correct. Both machines
need touching and naming one of them sends a fitter somewhere useful; naming a
cause that is not happening does not. Saying "none" to a run where two machines
are drifting is the failure worth catching.

    python scripts/eval_hard_drift.py
    python scripts/eval_hard_drift.py --per-family 6 --no-edge
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402
from cell.drift import (  # noqa: E402
    VISIBLE_RATIO,
    build_hard_case,
    discriminators,
    prompt_for,
    user_prompt,
)

CLOUD_MODEL = "nvidia/nemotron-3-super-120b-a12b"
ADAPTER = "data/finetune/adapter"
BASE_MODEL = "Qwen/Qwen3-1.7B"

# family -> the scenario injected for it
PLAN = [
    ("clear", "press"), ("clear", "fixture"), ("clear", "camera"),
    ("marginal", "press"), ("marginal", "fixture"),
    ("lucky_noise", "none"),
    ("subtle_camera", "camera"),
    ("mixed", "mixed"),
]

THRESHOLD_NOTE = f"""

The cell fits a least squares line to each channel and measures its slope against
its own residual scatter. A channel under {VISIBLE_RATIO:.0f} times that scatter is
not trending however large the slope looks; over {VISIBLE_RATIO:.0f} times it is.
Some runs sit close to that line, and some have more than one machine going out at
once. Say what the evidence supports."""


def stats_block(rows: list[dict]) -> str:
    d = discriminators(rows)
    out = ["", "Fitted over the whole run:",
           f"  tilt       {d['tilt_slope']:+.4f} mm/part   {d['tilt_ratio']:5.2f} times scatter",
           f"  bottom     {d['shift_slope']:+.4f} mm/part   {d['shift_ratio']:5.2f} times scatter"]
    if "reference_ratio" in d:
        out.append(f"  reference  {d['reference_slope']:+.4f} mm/part   "
                   f"{d['reference_ratio']:5.2f} times scatter")
    return "\n".join(out)


def rule(rows: list[dict]) -> str:
    """The whole diagnosis as arithmetic, exactly as a threshold would do it."""
    d = discriminators(rows)
    tilt = d["tilt_ratio"] >= VISIBLE_RATIO
    shift = d["shift_ratio"] >= VISIBLE_RATIO
    ref = d.get("reference_ratio", 0.0) >= VISIBLE_RATIO
    if ref:
        return "camera"
    if tilt and not shift:
        return "press"
    if shift and not tilt:
        return "fixture"
    return "none"


def is_correct(said: str | None, truth: str) -> bool:
    """Mixed runs accept either real fault; everything else is exact."""
    if truth == "mixed":
        return said in ("press", "fixture")
    return said == truth


def ask_cloud(rows: list[dict], model: str, tell_threshold: bool) -> tuple[str, float]:
    client = nebius.client().with_options(timeout=90.0)
    system = prompt_for(rows) + (THRESHOLD_NOTE if tell_threshold else "")
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=500, reasoning_effort="none",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_prompt(rows) + stats_block(rows)}],
        )
    except Exception as exc:  # noqa: BLE001
        return f"error: {type(exc).__name__}", (time.perf_counter() - started) * 1000
    raw = nebius.reply_text(response.choices[0].message) or ""
    ms = (time.perf_counter() - started) * 1000
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return str(json.loads(raw[start:end + 1]).get("cause")), ms
    except (ValueError, json.JSONDecodeError):
        return "unparsed", ms


def load_edge(base: str, adapter: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return tokenizer, model


def ask_edge(rows: list[dict], tokenizer, model) -> tuple[str, float]:
    import torch

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": prompt_for(rows)},
         {"role": "user", "content": user_prompt(rows)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    ms = (time.perf_counter() - started) * 1000
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    start, end = raw.find("{"), raw.rfind("}")
    try:
        return str(json.loads(raw[start:end + 1]).get("cause")), ms
    except (ValueError, json.JSONDecodeError):
        return "unparsed", ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--parts", type=int, default=30)
    parser.add_argument("--model", default=CLOUD_MODEL)
    parser.add_argument("--no-edge", action="store_true")
    parser.add_argument("--no-cloud", action="store_true")
    parser.add_argument("--out", default="data/hard_drift.jsonl")
    args = parser.parse_args()

    cases = []
    for family, scenario in PLAN:
        found = 0
        for seed in range(3, 3 + args.per_family * 12):
            if found >= args.per_family:
                break
            got = build_hard_case(family, scenario, seed, parts=args.parts)
            if got is None:
                continue
            rows, truth, d = got
            cases.append({"family": family, "truth": truth, "seed": seed,
                          "rows": rows, "ratios": d})
            found += 1
    print(f"{len(cases)} cases across {len(PLAN)} family and scenario pairs, "
          f"built to sit on the boundary.\n")

    edge = None
    if not args.no_edge and Path(ADAPTER).is_dir():
        print("loading the edge model ...", flush=True)
        edge = load_edge(BASE_MODEL, ADAPTER)
        warm = cases[0]["rows"]
        ask_edge(warm, *edge)          # warm the GPU before anything is timed

    arms: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        rows, truth = case["rows"], case["truth"]
        said_rule = rule(rows)
        arms["rule"].append({**case, "said": said_rule,
                             "correct": is_correct(said_rule, truth), "ms": 0.0})
        if not args.no_cloud:
            for arm, tell in (("cloud", True), ("cloud_blind", False)):
                said, ms = ask_cloud(rows, args.model, tell)
                arms[arm].append({**case, "said": said,
                                  "correct": is_correct(said, truth), "ms": ms})
        if edge is not None:
            said, ms = ask_edge(rows, *edge)
            arms["edge"].append({**case, "said": said,
                                 "correct": is_correct(said, truth), "ms": ms})

    families = [f for f, _ in PLAN]
    seen: list[str] = []
    for f in families:
        if f not in seen:
            seen.append(f)
    print(f"{'arm':>12} {'total':>8} " + " ".join(f"{f[:13]:>14}" for f in seen)
          + f" {'median ms':>10}")
    print("-" * (22 + 15 * len(seen) + 11))

    records = []
    for arm, rows_ in arms.items():
        cells = []
        for f in seen:
            got = [r for r in rows_ if r["family"] == f]
            cells.append(f"{sum(1 for r in got if r['correct'])}/{len(got)}"
                         if got else "-")
        ms = statistics.median(r["ms"] for r in rows_) if rows_ else 0
        print(f"{arm:>12} {sum(1 for r in rows_ if r['correct']):>4}/{len(rows_):<3} "
              + " ".join(f"{c:>14}" for c in cells) + f" {ms:>10.0f}")
        for r in rows_:
            records.append({k: v for k, v in r.items() if k != "rows"}
                           | {"arm": arm})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
