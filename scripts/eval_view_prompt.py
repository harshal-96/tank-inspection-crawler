"""Were the larger models over-cautious about view planning, or obeying the prompt?

The routing study scores Super 120B at 0 of 39 and Lightning at 1 of 39, and this
project explained that as larger models being inherently more cautious. That
explanation does not survive reading the prompt they were given, which says:

    "Accept a single view only if its signals are excellent and no other view is
     available."

Every part in the campaign starts with exactly one view, and "excellent" is never
defined anywhere. Read literally, that sentence instructs a model to ask for a
second look on every part in the set. Super asked for another look 39 times out of
39, which is not obviously bad judgment. It may be obedience to an instruction with
no number in it.

So the same decisions are put to the same models twice: the prompt as written, and
the prompt with one sentence added that says when a single view is already enough.
The threshold is taken from what the cell already treats as a good view elsewhere in
the codebase rather than tuned against the answers.

The cases, the call path and the scoring are borrowed from the routing study rather
than rebuilt here. The first version of this script rebuilt them, and produced a
number that contradicted it: Lightning scored 1 of 39 there and 18 of 20 here. That
was a difference between two harnesses, not between two prompts. Nothing is
comparable unless the sentence under test is the only thing that changes.

    python scripts/eval_view_prompt.py --cases 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain.viewplanner import plan_next_view  # noqa: E402
from scripts.eval_router_truth import decision_points  # noqa: E402

MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3_5-Lightning",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
]

# The boundary the prompt never gave.
THRESHOLD = """

A single view is already enough, and you should accept it, when all of these hold:
included_angle_deg is within 5 of 90, inlier_balance is at least 0.60, both fit RMS
values are under 0.5 mm, and path_length_mm is at least 300. Asking for another
look costs cycle time, so ask only when one of those is actually violated."""


def ask(point: dict, model: str, with_threshold: bool) -> tuple[bool, bool, float]:
    """One decision, through the planner the cell itself calls.

    Returns whether it accepted, whether that was the right call against ground
    truth, and how long it took.
    """
    started = time.perf_counter()
    try:
        decision = plan_next_view(
            point["evidence"], point["remaining"], model=model,
            system_extra=THRESHOLD if with_threshold else "")
    except Exception:  # noqa: BLE001
        return False, False, (time.perf_counter() - started) * 1000
    ms = (time.perf_counter() - started) * 1000
    said_look_again = not decision.accept
    return decision.accept, said_look_again == point["should_look_again"], ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="data/campaign50.jsonl")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--out", default="data/view_prompt.jsonl")
    args = parser.parse_args()

    points = list(decision_points(Path(args.campaign)))[:args.cases]
    worth = sum(1 for p in points if p["should_look_again"])
    print(f"{len(points)} decision points from the routing study, scored the same way.\n"
          f"Another look was actually worth taking on {worth} of them.\n")
    print(f"{'model':>30} {'prompt':>22} {'correct':>10} {'accepted':>10}")
    print("-" * 76)

    records = []
    for model in MODELS:
        for with_threshold in (False, True):
            correct = accepted = 0
            for point in points:
                took, ok, ms = ask(point, model, with_threshold)
                correct += ok
                accepted += took
                records.append({"model": model, "threshold_stated": with_threshold,
                                "trial": point["trial"], "accept": took,
                                "correct": ok, "latency_ms": round(ms)})
            label = "threshold stated" if with_threshold else "as written"
            print(f"{model.split('/')[-1][:30]:>30} {label:>22} "
                  f"{correct:>6}/{len(points):<3} {accepted:>6}/{len(points):<3}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
