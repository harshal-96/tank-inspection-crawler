"""Which model on this account is actually best at the diagnosis?

The routing study compared four Nemotron sizes and found the obvious answer wrong.
This asks the wider question: of everything Token Factory serves, which model reads
a run of parts best?

Published reasoning benchmarks are a poor guide here and were checked first. MMLU
and GPQA measure recall and multi-step argument over text. This task is neither. It
is: read thirty rows of noisy millimetres, work out which of three columns carries
a trend that clears its own scatter, and refuse to name a fault when none does. A
model can be excellent at graduate physics and bad at that.

So every candidate gets the same four-way question, the same four held-out seeds,
the same prompt, and the same parser. Two things are recorded besides the score,
because the score alone has misled this project twice already:

  degenerate    a model that answers the same cause every time scores 4 of 16 by
                arithmetic. That is not a diagnosis and is reported separately.

  the pair      fixture and camera are identical in the seam and separated only by
                the reference column. Getting those right is the part that needs
                reading rather than guessing.

    python scripts/eval_model_field.py
    python scripts/eval_model_field.py --models nvidia/nemotron-3-super-120b-a12b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402
from cell.drift import (  # noqa: E402
    SCENARIOS_WITH_CAMERA,
    build_stream,
    prompt_for,
    quiet_seed,
    user_prompt,
)

# Everything on the account that is a plausible candidate: text capable, and large
# enough to be worth the call. The embedding model and the two vision models are
# left out because they cannot do this, not because they were not considered.
CANDIDATES = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/Nemotron-3_5-Lightning",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "moonshotai/Kimi-K3",
    "zai-org/GLM-5.3",
    "Qwen/Qwen3.5-397B-A17B",
    "openai/gpt-oss-120b",
    "NousResearch/Hermes-4-405B",
    "MiniMaxAI/MiniMax-M3",
]

CONFUSABLE = ("fixture", "camera")


def ask(rows: list[dict], model: str) -> dict:
    """One question, one model.

    reasoning_effort is a Nemotron parameter and other families reject it, so the
    call is retried without it rather than scoring a model zero for refusing an
    argument it never claimed to take.
    """
    client = nebius.client()
    messages = [
        {"role": "system", "content": prompt_for(rows)},
        {"role": "user", "content": user_prompt(rows)},
    ]
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=400,
            reasoning_effort="none", messages=messages,
        )
    except Exception:  # noqa: BLE001
        try:
            response = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=700, messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            return {"cause": "error", "confidence": 0.0,
                    "evidence": f"{type(exc).__name__}: {exc}"[:200],
                    "latency_ms": round((time.perf_counter() - started) * 1000)}

    raw = nebius.reply_text(response.choices[0].message) or ""
    latency = (time.perf_counter() - started) * 1000
    start, end = raw.find("{"), raw.rfind("}")
    try:
        data = json.loads(raw[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        data = {"cause": "unparsed", "confidence": 0.0, "evidence": raw[:160]}
    data["latency_ms"] = round(latency)
    return data


def safe(text: object) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def run(model: str, parts: int, seeds: list[int]) -> dict:
    records = []
    for seed in seeds:
        for scenario in SCENARIOS_WITH_CAMERA:
            rows, truth = build_stream(parts, scenario, seed, reference=True)
            verdict = ask(rows, model)
            records.append({
                "model": model, "scenario": truth, "seed": seed,
                "said": verdict.get("cause"), "correct": verdict.get("cause") == truth,
                "confidence": verdict.get("confidence"),
                "latency_ms": verdict.get("latency_ms"),
                "evidence": safe(verdict.get("evidence")),
            })

    said = Counter(r["said"] for r in records)
    confusable = [r for r in records if r["scenario"] in CONFUSABLE]
    errors = sum(1 for r in records if r["said"] in ("error", "unparsed"))
    return {
        "model": model,
        "n": len(records),
        "correct": sum(1 for r in records if r["correct"]),
        "confusable_correct": sum(1 for r in confusable if r["correct"]),
        "confusable_n": len(confusable),
        "degenerate": len({r["said"] for r in records}) == 1,
        "always_said": said.most_common(1)[0][0] if len(said) == 1 else None,
        "distinct_answers": len(said),
        "unusable": errors,
        "median_latency_ms": statistics.median(r["latency_ms"] for r in records),
        "total_latency_s": sum(r["latency_ms"] for r in records) / 1000.0,
        "cases": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, default=30)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default="data/model_field.jsonl")
    args = parser.parse_args()

    models = args.models or CANDIDATES
    seeds = [quiet_seed(args.parts, 100 + run * 137) for run in range(args.runs)]

    print(f"{len(models)} models, {len(seeds) * len(SCENARIOS_WITH_CAMERA)} cases each, "
          f"four causes with the reference column.\n")
    print(f"{'model':>38} {'score':>7} {'pair':>6} {'answers':>8} {'median ms':>10}  note")
    print("-" * 92)

    results = []
    for model in models:
        result = run(model, args.parts, seeds)
        results.append(result)
        note = ""
        if result["unusable"]:
            note = f"{result['unusable']} replies unusable"
        elif result["degenerate"]:
            note = f"said {result['always_said']!r} every time"
        print(f"{model[-38:]:>38} {result['correct']:>3}/{result['n']:<3} "
              f"{result['confusable_correct']:>2}/{result['confusable_n']:<3} "
              f"{result['distinct_answers']:>8} {result['median_latency_ms']:>10.0f}  {note}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({k: v for k, v in result.items() if k != "cases"}) + "\n")
    with out.with_name(out.stem + "_cases.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for case in result["cases"]:
                handle.write(json.dumps(case) + "\n")

    usable = [r for r in results if not r["degenerate"] and not r["unusable"]]
    if usable:
        best = max(usable, key=lambda r: (r["correct"], -r["median_latency_ms"]))
        print(f"\nbest that is actually diagnosing: {best['model']} "
              f"at {best['correct']}/{best['n']}, median {best['median_latency_ms']:.0f} ms")
    degenerate = [r["model"] for r in results if r["degenerate"]]
    if degenerate:
        print(f"answered one cause to everything: {', '.join(degenerate)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
