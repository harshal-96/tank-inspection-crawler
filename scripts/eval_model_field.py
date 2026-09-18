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
    parse_verdict,
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

# A call that never returns is worse than a model that answers badly, because it
# stops the comparison rather than losing it. The SDK default is ten minutes, and
# one queued model behind it stalled a twelve-model run for a quarter of an hour
# with nothing to show. Bounded here, and a model that keeps timing out is dropped
# rather than allowed to hold up the rest.
CALL_TIMEOUT_S = 45.0
GIVE_UP_AFTER = 3

# Room for a model that reasons before it answers. Four hundred is plenty for the
# JSON and nowhere near enough for the thinking some families do first.
THINKING_TOKENS = 3000


def ask(rows: list[dict], model: str, timeout_s: float = CALL_TIMEOUT_S) -> dict:
    """One question, one model.

    reasoning_effort is a Nemotron parameter and other families reject it, so the
    call is retried without it rather than scoring a model zero for refusing an
    argument it never claimed to take.
    """
    messages = [
        {"role": "system", "content": prompt_for(rows)},
        {"role": "user", "content": user_prompt(rows)},
    ]
    started = time.perf_counter()
    # One policy for every model, the same one the cell runs: a rejected option
    # is dropped and the call made again, and a reply that spent its budget
    # thinking is asked once more with room. The first version of this file had
    # its own three-stage version of the same idea, which is how two harnesses
    # come to disagree.
    try:
        reply = nebius.chat(messages, model=model, max_tokens=400, reasoning_effort="none",
                            parses=nebius.extract_json,
                            api=nebius.client().with_options(timeout=timeout_s))
    except Exception as exc:  # noqa: BLE001
        return {"cause": "error", "confidence": 0.0,
                "evidence": f"{type(exc).__name__}: {exc}"[:200],
                "latency_ms": round((time.perf_counter() - started) * 1000)}
    raw = reply.raw or ""

    latency = (time.perf_counter() - started) * 1000
    data = parse_verdict(raw)
    data["latency_ms"] = round(latency)
    return data


def _has_json(raw: str) -> bool:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        json.loads(raw[start:end + 1])
        return True
    except (ValueError, json.JSONDecodeError):
        return False


def safe(text: object) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def run(model: str, parts: int, seeds: list[int]) -> dict:
    records = []
    consecutive_failures = 0
    for seed in seeds:
        for scenario in SCENARIOS_WITH_CAMERA:
            if consecutive_failures >= GIVE_UP_AFTER:
                break
            rows, truth = build_stream(parts, scenario, seed, reference=True)
            verdict = ask(rows, model)
            if verdict.get("cause") == "error":
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            records.append({
                "model": model, "scenario": truth, "seed": seed,
                "said": verdict.get("cause"), "correct": verdict.get("cause") == truth,
                "confidence": verdict.get("confidence"),
                "latency_ms": verdict.get("latency_ms"),
                "evidence": safe(verdict.get("evidence")),
            })

    said = Counter(r["said"] for r in records)
    confusable = [r for r in records if r["scenario"] in CONFUSABLE]
    errors = sum(1 for r in records
                 if r["said"] in ("error", "unparsed", "out_of_vocabulary"))

    # A reply nobody could parse is not a wrong answer, and scoring it as one
    # misrepresents the model rather than the harness. gpt-oss answered four cases
    # and got all four right; counting its twelve truncated replies as failures
    # published it at 4 of 16, which is a false claim about a model that was
    # correct every time it managed to speak. Both numbers are kept: the raw one
    # is what the cell would actually get, and the other is what the model knows.
    unusable = ("error", "unparsed", "out_of_vocabulary")
    answered = [r for r in records if r["said"] not in unusable]
    answered_confusable = [r for r in confusable if r["said"] not in unusable]
    return {
        "model": model,
        "n": len(records),
        "correct": sum(1 for r in records if r["correct"]),
        "answered": len(answered),
        "correct_of_answered": sum(1 for r in answered if r["correct"]),
        "confusable_answered": len(answered_confusable),
        "correct_confusable_of_answered": sum(
            1 for r in answered_confusable if r["correct"]),
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
    print(f"{'model':>38} {'raw':>7} {'of answered':>12} {'pair':>6} "
          f"{'answers':>8} {'median ms':>10}  note")
    print("-" * 104)

    results = []
    for model in models:
        started = time.perf_counter()
        result = run(model, args.parts, seeds)
        result["wall_s"] = round(time.perf_counter() - started, 1)
        results.append(result)
        note = ""
        if result["unusable"]:
            note = f"{result['unusable']} replies unusable"
        elif result["degenerate"]:
            note = f"said {result['always_said']!r} every time"
        of_answered = (f"{result['correct_of_answered']}/{result['answered']}"
                       if result["answered"] else "-")
        print(f"{model[-38:]:>38} {result['correct']:>3}/{result['n']:<3} "
              f"{of_answered:>12} "
              f"{result['correct_confusable_of_answered']:>2}/{result['confusable_answered']:<3} "
              f"{result['distinct_answers']:>8} {result['median_latency_ms']:>10.0f}  {note}",
              flush=True)

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
        best = max(usable, key=lambda r: (r["correct_of_answered"] / max(1, r["answered"]),
                                          -r["median_latency_ms"]))
        print(f"\nbest that is actually diagnosing: {best['model']} "
              f"at {best['correct']}/{best['n']}, median {best['median_latency_ms']:.0f} ms")
    degenerate = [r["model"] for r in results if r["degenerate"]]
    if degenerate:
        print(f"answered one cause to everything: {', '.join(degenerate)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
