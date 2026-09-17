"""Score every model tier on whether its decisions were RIGHT, not on whether the
biggest model agreed with them.

The first attempt at this compared each tier against the largest model in the
family and reported that the small one agreed with it 0 percent of the time. That
was a broken measurement, and it is worth saying why, because the failure is easy
to repeat.

On a clean single view, the small model says accept and the largest says look
again. Against ground truth, accepting is right: one good view lands 0.907 mm from
the true joint and four looks land 0.886 mm, so the extra looks buy two hundredths
of a millimetre for three more robot moves. The large model is not more correct
there, it is more cautious, and caution priced in robot moves is a cost like any
other. Treating it as the reference measured conformity, not quality.

So this scores against the thing that actually settles it. Every decision point is
replayed, and because the simulation knows where the joint really is, both branches
can be evaluated: what the answer would be if the cell accepted now, and what it
would be if it took one more look. A decision is correct when it picks the branch
that a shop would pick, which means accepting unless another look is worth the
move it costs.

    python scripts/eval_router_truth.py --campaign data/campaign50.jsonl
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

from brain.router import LADDER, assess, tier_of  # noqa: E402
from brain.viewplanner import plan_next_view  # noqa: E402

VIEWS = [-45.0, -22.5, 0.0, 22.5, 45.0]

# How much a further look has to be worth before it earns the robot move it costs.
# Two tenths of a millimetre on a joint that has to land inside a millimetre: less
# than this and the move is not buying anything a weld can feel.
WORTH_ANOTHER_LOOK_MM = 0.20


def fused_error_mm(looks: list[dict], truth: list[float]) -> float | None:
    """Median of the accepted estimates, scored against the true joint."""
    points = [lk["joint_mm"] for lk in looks if lk.get("joint_mm")]
    if not points:
        return None
    x = statistics.median(p[0] for p in points) / 1000.0
    y = statistics.median(p[1] for p in points) / 1000.0
    return ((x - truth[0]) ** 2 + (y - truth[1]) ** 2) ** 0.5 * 1000.0


def decision_points(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        looks, truth = record["looks"], record["true_joint"]
        for taken in range(1, len(looks)):
            evidence = [{
                "view_deg": lk["view_deg"],
                "joint_mm": lk.get("joint_mm"),
                "failed": lk.get("failed"),
                "signals": lk.get("signals") or {},
            } for lk in looks[:taken]]
            seen = {e["view_deg"] for e in evidence}
            remaining = [v for v in VIEWS if v not in seen]
            if not remaining:
                continue

            now = fused_error_mm(looks[:taken], truth)
            after = fused_error_mm(looks[:taken + 1], truth)
            if now is None or after is None:
                continue
            gain = now - after
            yield {
                "trial": record["trial"],
                "evidence": evidence,
                "remaining": remaining,
                "error_if_accept_mm": now,
                "error_if_look_again_mm": after,
                "gain_mm": gain,
                "should_look_again": gain > WORTH_ANOTHER_LOOK_MM,
            }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="data/campaign50.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="data/router_truth_eval.jsonl")
    args = parser.parse_args()

    points = list(decision_points(Path(args.campaign)))
    if args.limit:
        points = points[: args.limit]

    arms = {tier_of(m): m for m in LADDER}
    arms["routed"] = None

    print(f"{len(points)} decision points where ground truth can settle the call")
    worth = sum(1 for p in points if p["should_look_again"])
    print(f"another look was actually worth taking on {worth} of them "
          f"({100 * worth / len(points):.0f}%)\n")

    started = time.time()
    results: dict[str, list[dict]] = {name: [] for name in arms}

    for point in points:
        for name, model in arms.items():
            route = assess(point["evidence"]) if model is None else None
            chosen = route.model if route else model
            decision = plan_next_view(point["evidence"], point["remaining"], model=chosen)
            said_look_again = not decision.accept
            results[name].append({
                "trial": point["trial"],
                "tier": route.tier if route else name,
                "said_look_again": said_look_again,
                "should_look_again": point["should_look_again"],
                "correct": said_look_again == point["should_look_again"],
                "gain_mm": point["gain_mm"],
                "cost_mm": point["gain_mm"] if (point["should_look_again"]
                                                and not said_look_again) else 0.0,
                "latency_ms": round(decision.latency_ms or 0.0),
                "tokens": (decision.prompt_tokens or 0) + (decision.completion_tokens or 0),
            })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for name, rows in results.items():
            for row in rows:
                handle.write(json.dumps({"arm": name, **row}) + "\n")

    print(f"{'arm':>12} {'correct':>9} {'extra looks':>12} {'missed gain':>12} "
          f"{'worst miss':>11} {'time':>9}")
    print("-" * 72)
    for name in ("nano", "lightning", "super", "ultra", "routed"):
        rows = results.get(name) or []
        if not rows:
            continue
        correct = sum(1 for r in rows if r["correct"])
        extra = sum(1 for r in rows if r["said_look_again"] and not r["should_look_again"])
        missed = sum(r["cost_mm"] for r in rows)
        worst = max((r["cost_mm"] for r in rows), default=0.0)
        seconds = sum(r["latency_ms"] for r in rows) / 1000.0
        print(f"{name:>12} {correct:>4}/{len(rows):<4} {extra:>12} "
              f"{missed:>10.2f} mm {worst:>8.2f} mm {seconds:>7.1f} s")

    routed_tiers = Counter(r["tier"] for r in results["routed"])
    print(f"\nrouted used: {dict(routed_tiers)}")
    print("extra looks are robot moves spent on a part that did not need them.")
    print("missed gain is accuracy given up by accepting when another look was worth it.")
    print(f"\nwrote {out} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
