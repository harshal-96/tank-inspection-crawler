"""Run the drift diagnosis on the cell's own GPU, and score it like the rest.

The point of the exercise. Four Nemotron sizes were asked, in the cloud, to read a
run of parts and name which machine is going out. The small ones answered "press"
every time, which scores 4 of 12 by arithmetic; only the 120B read the evidence.
That put the one question a shop actually pays for behind a large model and a
network connection.

This asks the same twelve runs of a 1.7B model that was trained on Token Factory
and now runs on the workstation beside the cell. Nothing about the question is
made easier for it:

  same generator     cell/drift.py builds the streams, the same function that
                     built the ones the Nemotron sizes were scored on.
  same seeds         the four the published evaluation landed on, which the
                     training set was explicitly generated away from.
  same prompt        the system prompt is imported, not retyped. A fine-tuned
                     model given a better prompt than its baselines has not been
                     compared to them.
  same scoring       the output file has the same shape as the cloud runs, so the
                     dashboard puts them on one axis without special cases.

    python scripts/eval_drift_edge.py
    python scripts/eval_drift_edge.py --base-only      # the untrained 1.7B, for contrast
    python scripts/eval_drift_edge.py --with-camera    # the four-way question

A run writes to runs/ (gitignored) with a timestamp in the name. The recorded
results the dashboard reads, data/drift_eval_edge*.jsonl, are never replaced by
a rerun unless --out names one of them and --force is given: latencies always
differ between runs and answers can, so a quiet overwrite would change published
evidence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cell.drift import (  # noqa: E402
    SCENARIOS,
    SCENARIOS_WITH_CAMERA,
    build_stream,
    parse_verdict,
    prompt_for,
    quiet_seed,
    trend,
    user_prompt,
)

BASE_MODEL = "Qwen/Qwen3-1.7B"
ADAPTER = "data/finetune/adapter"
# The adapter is about 81 MB of weights and is not in git. It is to be attached
# to a release of the repository as a zip; this is that asset's download link,
# and it stays empty until the release exists rather than pointing at nothing.
ADAPTER_RELEASE_URL = ""
MAX_NEW_TOKENS = 220


def load(base: str, adapter: str | None):
    """Base model on the GPU, with the trained adapter on top of it if there is one."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {base} ...")
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    if adapter:
        from peft import PeftModel

        print(f"applying adapter {adapter} ...")
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()     # merged, so inference is plain forward passes
    model.eval()
    device = next(model.parameters()).device
    print(f"ready on {device}")
    return tokenizer, model


def ask(tokenizer, model, rows: list[dict]) -> dict:
    import torch

    messages = [
        # Paired to the rows, so a model is never told about a reference column
        # that is not in its table or shown one nobody described.
        {"role": "system", "content": prompt_for(rows)},
        {"role": "user", "content": user_prompt(rows)},
    ]
    # Qwen3 thinks by default. The cloud baselines were all run with reasoning
    # turned off, so this is turned off too rather than being given the advantage.
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.perf_counter() - started) * 1000
    raw = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:],
                           skip_special_tokens=True)

    data = parse_verdict(raw)
    data["latency_ms"] = round(latency)
    return data


def safe(text: object) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def protected_paths() -> set[Path]:
    """The recorded edge results, which a new run must not quietly replace."""
    return {p.resolve() for p in (ROOT / "data").glob("drift_eval_edge*.jsonl")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=30)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--base", default=BASE_MODEL)
    parser.add_argument("--adapter", default=ADAPTER)
    parser.add_argument("--base-only", action="store_true",
                        help="Score the untrained model, to show what training bought")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--out", default=None,
                        help="Where to write. Default: runs/<stem>_<timestamp>.jsonl")
    parser.add_argument("--force", action="store_true",
                        help="Allow --out to replace a recorded data/drift_eval_edge*.jsonl")
    parser.add_argument("--with-camera", action="store_true",
                        help="The four-way question, including a drifting camera, with "
                             "the reference feature that separates it from a drifting "
                             "fixture. Harder, and the one that matters.")
    args = parser.parse_args()

    scenarios = SCENARIOS_WITH_CAMERA if args.with_camera else SCENARIOS

    adapter = None if args.base_only else args.adapter
    if adapter and not Path(adapter).is_dir():
        where = ADAPTER_RELEASE_URL or "no release yet; the weights are to be attached to one"
        raise SystemExit(
            f"no adapter at {adapter}. The trained weights are not in git.\n"
            f"  download: {where}\n"
            f"  unzip it so that {adapter}/adapter_model.safetensors exists, or, with the\n"
            f"  Token Factory key that trained it, run scripts/finetune_drift.py --fetch\n"
            f"  (--base-only runs the untrained control without it)")

    stem = "drift_eval_edge4" if args.with_camera else "drift_eval_edge"
    stem += "_base" if args.base_only else ""
    if args.out:
        out = Path(args.out)
    else:
        out = ROOT / "runs" / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    # Checked before the model loads, so a refused run costs nothing.
    if out.resolve() in protected_paths() and not args.force:
        raise SystemExit(
            f"refusing to write over {out}: it is a recorded result. Write elsewhere,\n"
            f"  or pass --force if replacing it is the intent.")

    tokenizer, model = load(args.base, adapter)

    # One throwaway inference before anything is timed. A laptop GPU starts at a
    # low clock and ramps, so the first calls in a run are several times slower
    # than the rest: the four-way evaluation opened at 39 s and settled at 9 s,
    # falling monotonically, which is warm-up rather than the question being
    # harder. Timing that and reporting the median describes the measurement
    # rather than the model.
    warm, _ = build_stream(args.parts, scenarios[0], 4242, reference=args.with_camera)
    print("warming the GPU ...", end=" ", flush=True)
    ask(tokenizer, model, warm)
    print("ready")

    records = []
    print(f"\n{'scenario':>9} {'seed':>5} {'tilt slope':>11} {'shift slope':>12} "
          f"{'said':>9} {'conf':>6} {'ok':>4} {'ms':>6}")
    print("-" * 74)

    for run in range(args.runs):
        for scenario in scenarios:
            seed = quiet_seed(args.parts, 100 + run * 137)
            rows, truth = build_stream(args.parts, scenario, seed, args.severity,
                                       reference=args.with_camera)
            verdict = ask(tokenizer, model, rows)
            correct = verdict.get("cause") == truth
            records.append({
                "scenario": truth, "seed": seed, "said": verdict.get("cause"),
                "confidence": verdict.get("confidence"), "correct": correct,
                "tilt_slope_mm_per_part": round(trend([r["tilt_mm"] for r in rows]), 4),
                "shift_slope_mm_per_part": round(
                    trend([r["seam_bottom_mm"] for r in rows]), 4),
                "evidence": safe(verdict.get("evidence")),
                "latency_ms": verdict.get("latency_ms"),
            })
            r = records[-1]
            print(f"{truth:>9} {seed:>5} {r['tilt_slope_mm_per_part']:>10.4f} "
                  f"{r['shift_slope_mm_per_part']:>11.4f} {str(r['said']):>9} "
                  f"{str(r['confidence']):>6} {'yes' if correct else 'NO':>4} "
                  f"{r['latency_ms']:>6}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    correct = sum(1 for r in records if r["correct"])
    print(f"\n{correct}/{len(records)} diagnosed correctly")
    for scenario in scenarios:
        rows = [r for r in records if r["scenario"] == scenario]
        print(f"  {scenario:<8} {sum(1 for r in rows if r['correct'])}/{len(rows)}")
    if args.with_camera:
        confusable = [r for r in records if r["scenario"] in ("fixture", "camera")]
        got = sum(1 for r in confusable if r["correct"])
        print(f"  the two that look alike on the seam alone: {got}/{len(confusable)}")
    said = {r["said"] for r in records}
    if len(said) == 1:
        print(f"  [warning] answered {said.pop()!r} every time, which is not diagnosis")
    print(f"  median latency {statistics.median(r['latency_ms'] for r in records):.0f} ms")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
