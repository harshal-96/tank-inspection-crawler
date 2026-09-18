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

import numpy as np  # noqa: E402

from brain.router import LADDER, assess, tier_of  # noqa: E402
from brain.viewplanner import plan_next_view  # noqa: E402
from scripts.run_campaign import load_context, take_look  # noqa: E402

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


def first_view_points(manifest_path: Path, worth_mm: float = WORTH_ANOTHER_LOOK_MM):
    """Every part seen first from every angle, with the truth of what a look buys.

    The replayed campaign started every part from the square-on view, and on
    that view another look was never worth taking: 0 of 39. A test where the
    right answer is always "accept" cannot catch a model that accepts blindly,
    and three sizes scored full marks on it. So each capture is also measured
    from each of its five angles as if that were the first look. A view that
    goes edge-on gives a poor estimate, and there a look is genuinely worth the
    move. The outcome of looking again is the average over the untried angles
    of what fusing one more view would give, so the truth does not depend on
    which angle any model would have picked.
    """
    manifest, ctx = load_context(manifest_path)
    for row in manifest["rows"]:
        truth = np.array(row["true_joint"], dtype=float)
        looks = [take_look(view, truth, ctx) for view in row["views"]]
        for first in looks:
            others = [lk for lk in looks if lk is not first and not lk.failed]
            if not others:
                continue
            # One more look means one more angle, and the cell's own habit is
            # to swing back toward square-on, so the look-again branch is the
            # untried angle nearest 0 that measured. Averaging over every other
            # angle, which the first draft did, let a 200 mm miss from the far
            # side drag the fused answer down and made a good view look like
            # it needed help.
            nearest = sorted(others, key=lambda lk: (abs(lk.view_deg), -lk.view_deg))[0]
            if first.failed:
                # Nothing was measured. Accepting here means welding on the taught
                # path, and the part's own blind error is what that costs; one
                # more look gets the measurement back.
                now = float(row["blind_error_mm"])
                after = nearest.error_mm
            else:
                now = first.error_mm
                # fused_error_mm takes the truth in metres, as the campaign file
                # stores it; the first run of this passed millimetres and called
                # a 70 mm miss "accept".
                after = fused_error_mm([first.for_planner(), nearest.for_planner()],
                                       list(truth))
            if now is None or after is None:
                continue
            gain = now - after
            # With no measurement in hand there is nothing to accept: the cell
            # cannot know the part happens to sit near the taught path, so the
            # right call is to look, whatever the blind error turns out to be.
            yield {
                "trial": f"{row['trial']}@{first.view_deg:g}",
                "evidence": [first.for_planner()],
                "remaining": [float(lk.view_deg) for lk in looks if lk is not first],
                "error_if_accept_mm": float(now),
                "error_if_look_again_mm": float(now - gain),
                "gain_mm": float(gain),
                "should_look_again": bool(first.failed) or bool(gain > worth_mm),
            }


def balanced(points: list[dict], per_side: int, seed: int = 0) -> list[dict]:
    """The same number of look-needed and accept cases, drawn evenly by part."""
    import random

    rng = random.Random(seed)
    look = [p for p in points if p["should_look_again"]]
    keep = [p for p in points if not p["should_look_again"]]
    rng.shuffle(look)
    rng.shuffle(keep)
    n = min(per_side, len(look), len(keep))
    return look[:n] + keep[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="data/campaign50.jsonl")
    parser.add_argument("--manifest", default="data/cap50/manifest.json",
                        help="Captures measured from every angle for the first-view points.")
    parser.add_argument("--points", choices=("replayed", "first-view", "both"), default="both")
    parser.add_argument("--per-side", type=int, default=30,
                        help="First-view points kept per answer, look and accept, so the test "
                             "can catch both kinds of mistake.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the decision points and report them; call no model.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="data/router_truth_eval.jsonl")
    args = parser.parse_args()

    points: list[dict] = []
    if args.points in ("replayed", "both"):
        points += list(decision_points(Path(args.campaign)))
    if args.points in ("first-view", "both"):
        synthetic = list(first_view_points(Path(args.manifest)))
        worth = sum(1 for p in synthetic if p["should_look_again"])
        failed = sum(1 for p in synthetic if p["evidence"][0].get("measurement_failed"))
        print(f"{len(synthetic)} first-view points measured, a look worth taking on {worth}, "
              f"of which {failed} because the first view failed to measure at all")
        points += balanced(synthetic, args.per_side)
    for point in points:
        point.setdefault("kind", "replayed" if "@" not in str(point["trial"]) else "first-view")
    # The first run of the first-view points died at the final write on a numpy
    # bool, after every call had been made. So every point is checked as JSON
    # before a model is called, and rows are written as they are produced.
    json.dumps(points)
    if args.dry_run:
        for kind in ("replayed", "first-view"):
            sub = [p for p in points if p["kind"] == kind]
            if sub:
                need = sum(1 for p in sub if p["should_look_again"])
                print(f"  {kind:<10} {len(sub):>3} points, a look needed on {need}")
        return 0
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
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("w", encoding="utf-8")

    for point in points:
        for name, model in arms.items():
            route = assess(point["evidence"]) if model is None else None
            chosen = route.model if route else model
            decision = plan_next_view(point["evidence"], point["remaining"], model=chosen)
            said_look_again = not decision.accept
            results[name].append({
                "trial": point["trial"],
                "kind": point["kind"],
                "tier": route.tier if route else name,
                "said_look_again": said_look_again,
                # What the model said, so a wrong call can be read rather than guessed at.
                "reason": (decision.reason or "")[:400],
                "parse_error": decision.parse_error,
                "should_look_again": point["should_look_again"],
                "correct": said_look_again == point["should_look_again"],
                "gain_mm": point["gain_mm"],
                "cost_mm": point["gain_mm"] if (point["should_look_again"]
                                                and not said_look_again) else 0.0,
                "latency_ms": round(decision.latency_ms or 0.0),
                "tokens": (decision.prompt_tokens or 0) + (decision.completion_tokens or 0),
            })
            handle.write(json.dumps({"arm": name, **results[name][-1]}) + "\n")
            handle.flush()
    handle.close()

    print(f"{'arm':>12} {'correct':>9} {'extra looks':>12} {'blind accepts':>13} "
          f"{'missed gain':>12} {'worst miss':>11} {'time':>9}")
    print("-" * 86)
    for name in ("nano", "lightning", "super", "ultra", "routed"):
        rows = results.get(name) or []
        if not rows:
            continue
        correct = sum(1 for r in rows if r["correct"])
        extra = sum(1 for r in rows if r["said_look_again"] and not r["should_look_again"])
        blind = sum(1 for r in rows if r["should_look_again"] and not r["said_look_again"])
        missed = sum(r["cost_mm"] for r in rows)
        worst = max((r["cost_mm"] for r in rows), default=0.0)
        seconds = sum(r["latency_ms"] for r in rows) / 1000.0
        print(f"{name:>12} {correct:>4}/{len(rows):<4} {extra:>12} {blind:>13} "
              f"{missed:>10.2f} mm {worst:>8.2f} mm {seconds:>7.1f} s")

    routed_tiers = Counter(r["tier"] for r in results["routed"])
    print(f"\nrouted used: {dict(routed_tiers)}")
    print("extra looks are robot moves spent on a part that did not need them.")
    print("blind accepts are parts welded on a view that a look would have fixed.")
    print("missed gain is accuracy given up by accepting when another look was worth it.")
    print(f"\nwrote {out} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
