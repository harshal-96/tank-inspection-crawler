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

The model is also scored against the cell's own limits applied by arithmetic
(brain.viewplanner.rule_decision), because that is the test the rest of the
project uses before it keeps a model anywhere. The rule arm is free: it calls no
model, reads the recorded answers in data/router_truth_eval.jsonl, rebuilds the
same decision points, and puts the rule and the smallest model side by side
with an exact McNemar test on the points where they differ.

    python scripts/eval_router_truth.py --rule

The 39 replayed points rebuild exactly from data/campaign50.jsonl. The 50
first-view points were measured on the first render of the fifty parts, which
is no longer on disk, so they are rebuilt from the re-render in data/cap50v3
(not in the repository). Two checks gate the comparison, and the script stops
if either fails: every rebuilt point must get the same answer key as the
recorded one, and no signal reading the models quoted in their reasons may move
the rule's verdict when it is put in place of the rebuilt value. Without the
re-render it scores the first-view points from the signals kept in
data/router_rule_eval.json; nothing is rebuilt then, so the first check only
shows the kept file came from the same rows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from brain.router import SIZES, assess, tier_of  # noqa: E402
from brain.viewplanner import SINGLE_VIEW_OK, plan_next_view, rule_decision  # noqa: E402
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


# The rule arm. Everything below calls no model.

RULE_RECORD = Path("data/router_rule_eval.json")
LIMIT_SIGNALS = ("included_angle_deg", "inlier_balance", "face_a_fit_rms_mm",
                 "face_b_fit_rms_mm", "path_length_mm")


def flatten(evidence: list[dict]) -> list[dict]:
    """The evidence in the shape rule_decision reads.

    The cell hands the planner each look with its signals nested under
    "signals" and a refusal under "measurement_failed" (Look.for_planner), but
    rule_decision reads the signals at the top of each look and a refusal under
    "failed". Called on the cell's own shape it finds no signals and asks for
    another look every time; main() prints how often. The nested copy is kept
    so a rule that reads either shape gives the same answer here.
    """
    out = []
    for look in evidence:
        flat = dict(look)
        flat.update(look.get("signals") or {})
        if look.get("measurement_failed"):
            flat["failed"] = look["measurement_failed"]
        out.append(flat)
    return out


def rule_says_look_again(evidence: list[dict], remaining: list[float]) -> bool:
    accept, _ = rule_decision(flatten(evidence), remaining)
    return not accept


def misses_text(evidence: list[dict]) -> str:
    """The limits each look misses. rule_decision accepts if any look meets
    them, so every look is listed, not only the latest."""
    if len(evidence) == 1:
        return ", ".join(limits_missed(evidence[0])) or "none"
    return "; ".join(f"{float(look['view_deg']):g} deg: "
                     f"{', '.join(limits_missed(look)) or 'none'}" for look in evidence)


def limits_missed(look: dict) -> list[str]:
    """Which of the single-view limits one look misses, for reading a disagreement."""
    if look.get("failed") or look.get("measurement_failed"):
        return ["no measurement"]
    s = look.get("signals") or look
    ok = SINGLE_VIEW_OK
    missed = []
    if abs(float(s["included_angle_deg"]) - 90.0) > ok["angle_tol_deg"]:
        missed.append(f"angle {s['included_angle_deg']}")
    if float(s["inlier_balance"]) < ok["min_balance"]:
        missed.append(f"balance {s['inlier_balance']}")
    for face in ("face_a_fit_rms_mm", "face_b_fit_rms_mm"):
        if float(s[face]) >= ok["max_rms_mm"]:
            missed.append(f"{face[:6]} rms {s[face]}")
    if float(s["path_length_mm"]) < ok["min_path_mm"]:
        missed.append(f"path {s['path_length_mm']}")
    return missed


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two sided exact McNemar on the points one arm got right and the other did not."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


_QUOTED = re.compile(r"(included_angle_deg|inlier_balance|face_a_fit_rms_mm|"
                     r"face_b_fit_rms_mm|path_length_mm|face_a|face_b)(\D{0,12}?)(\d+\.\d+|\d+)")

# A reason quotes the prompt's limits as well as the readings: "exceeds 0.6",
# "is under 0.5", "is within 5 of 90". A number after one of these words is the
# limit being compared against, not what the model was shown. Dropping every
# value equal to a limit instead would also drop the angle readings of 90.0,
# which are real readings and agree with the rebuilt ones.
_COMPARED = re.compile(r"\b(at least|at most|within|exceeds?|under|over|above|below|"
                       r"less than|more than|greater than)\b")


def quoted_readings(point: dict, reasons: list[str]) -> list[dict]:
    """The signal readings the models quoted for one point, each tried on its own.

    The recorded rows keep each model's reason, and most reasons quote the
    signals they were shown. Each quoted reading is put in place of the rebuilt
    value, one at a time, and the rule asked again, so a single reading that
    would have decided the point differently cannot hide behind the others.
    """
    look = point["evidence"][0]
    if "signals" not in look:
        return []
    before = rule_says_look_again(point["evidence"], point["remaining"])
    out = []
    for reason in reasons:
        for name, gap, value in _QUOTED.findall(reason):
            if _COMPARED.search(gap):
                continue
            name = {"face_a": "face_a_fit_rms_mm", "face_b": "face_b_fit_rms_mm"}.get(name, name)
            here = float(look["signals"][name])
            seen = {**look, "signals": {**look["signals"], name: float(value)}}
            out.append({"trial": point["trial"], "signal": name, "quoted": float(value),
                        "here": here,
                        "moves": rule_says_look_again([seen], point["remaining"]) != before})
    return out


def run_rule_arm(args) -> int:
    """Score the arithmetic rule on the decision points the models were scored on."""
    record = Path(args.record)
    rows = [json.loads(line) for line in record.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)

    print(f"recorded answers in {record}, scored against the recorded answer key:")
    recorded = {}
    for name in ("nano", "lightning", "super", "ultra", "routed"):
        arm = by_arm.get(name) or []
        right = sum(1 for r in arm if r["said_look_again"] == r["should_look_again"])
        recorded[name] = [right, len(arm)]
        print(f"  {name:>10} {right} of {len(arm)}")

    nano = by_arm["nano"]
    replayed_rows = [r for r in nano if r["kind"] == "replayed"]
    first_rows = [r for r in nano if r["kind"] == "first-view"]

    # Replayed points come back in the order they were scored; check each one
    # is the same point before trusting the pairing.
    replayed = list(decision_points(Path(args.campaign)))
    if len(replayed) != len(replayed_rows):
        raise SystemExit(f"{len(replayed)} replayed points rebuilt, {len(replayed_rows)} recorded")
    for point, row in zip(replayed, replayed_rows, strict=True):
        if (point["trial"] != row["trial"] or abs(point["gain_mm"] - row["gain_mm"]) > 1e-9
                or point["should_look_again"] != row["should_look_again"]):
            raise SystemExit(f"replayed point {point['trial']} does not match the record")

    reasons: dict[str, list[str]] = {}
    for row in rows:
        if row["kind"] == "first-view":
            reasons.setdefault(row["trial"], []).append(row["reason"] or "")

    manifest = Path(args.manifest)
    stored = None
    if manifest.exists():
        rebuilt = {p["trial"]: p for p in first_view_points(manifest)}
        first = []
        for row in first_rows:
            if row["trial"] not in rebuilt:
                raise SystemExit(f"first-view point {row['trial']} cannot be rebuilt "
                                 f"from {manifest}")
            first.append(rebuilt[row["trial"]])
        source = f"rebuilt from {manifest}"
    else:
        rule_record = Path(args.rule_out)
        if not rule_record.exists():
            raise SystemExit(f"{manifest} is not here and neither is {rule_record}; "
                             "the first-view points need one of them")
        stored = json.loads(rule_record.read_text(encoding="utf-8"))
        first = [{"trial": p["trial"], "remaining": p["remaining"],
                  "should_look_again": p["should_look_again_rebuilt"],
                  "evidence": [{"view_deg": p["view_deg"], "measurement_failed": "refused"}
                               if p["failed"] else
                               {"view_deg": p["view_deg"], "signals": p["signals"]}]}
                 for p in stored["first_view_points"]]
        if [p["trial"] for p in first] != [r["trial"] for r in first_rows]:
            raise SystemExit(f"{rule_record} does not hold the recorded first-view points")
        source = f"read from the signals kept in {rule_record} ({manifest} is not here)"

    key_agrees = sum(1 for p, r in zip(first, first_rows, strict=True)
                     if p["should_look_again"] == r["should_look_again"])
    quoted = [q for p, r in zip(first, first_rows, strict=True)
              for q in quoted_readings(p, reasons.get(r["trial"], []))]
    distinct = {(q["trial"], q["signal"], q["quoted"]): q for q in quoted}
    moved = sorted({q["trial"] for q in quoted if q["moves"]})
    # Rounding a reading to two places moves it by well under 5 percent; a
    # bigger gap means the render the models saw measured something else.
    differ = [q for q in distinct.values()
              if abs(q["quoted"] - q["here"]) > 0.05 * max(abs(q["here"]), 1e-9)]
    widest = {}
    for q in distinct.values():
        best = widest.get(q["signal"])
        if best is None or abs(q["quoted"] - q["here"]) > abs(best["quoted"] - best["here"]):
            widest[q["signal"]] = q

    def score(points, recorded_rows):
        out = []
        for point, row in zip(points, recorded_rows, strict=True):
            out.append({
                "trial": row["trial"],
                "kind": row["kind"],
                "should_look_again": row["should_look_again"],
                "rule_look_again": rule_says_look_again(point["evidence"], point["remaining"]),
                "nano_look_again": row["said_look_again"],
                "literal_look_again": not rule_decision(point["evidence"], point["remaining"])[0],
                "gain_mm": row["gain_mm"],
                "missed": misses_text(point["evidence"]),
                "nano_reason": row["reason"] or "",
            })
        return out

    scored = score(replayed, replayed_rows) + score(first, first_rows)

    def tally(points):
        rule = sum(1 for p in points if p["rule_look_again"] == p["should_look_again"])
        model = sum(1 for p in points if p["nano_look_again"] == p["should_look_again"])
        only_rule = sum(1 for p in points if p["rule_look_again"] == p["should_look_again"]
                        and p["nano_look_again"] != p["should_look_again"])
        only_nano = sum(1 for p in points if p["nano_look_again"] == p["should_look_again"]
                        and p["rule_look_again"] != p["should_look_again"])
        return {"points": len(points), "rule_right": rule, "nano_right": model,
                "rule_only": only_rule, "nano_only": only_nano,
                "mcnemar_p": round(mcnemar_exact(only_rule, only_nano), 4)}

    sets = {
        "replayed": tally([p for p in scored if p["kind"] == "replayed"]),
        "first-view": tally([p for p in scored if p["kind"] == "first-view"]),
        "all": tally(scored),
    }

    print(f"\nreplayed points: {len(replayed)}, rebuilt exactly from {args.campaign}")
    print(f"first-view points: {len(first)}, {source}")
    here = "rebuilt" if stored is None else "kept"
    if stored is None:
        print(f"  answer key on the rebuilt points equals the recorded key on {key_agrees} "
              f"of {len(first)}")
    else:
        print(f"  answer key kept in {args.rule_out} equals the recorded key on {key_agrees} "
              f"of {len(first)}; the points were not rebuilt here")
    print(f"  {len(quoted)} signal readings quoted in the models' reasons ({len(distinct)} "
          f"distinct), each put in place of the {here} value one at a time: the rule's "
          f"verdict moves on {len(moved)} points")
    print(f"  {len(differ)} distinct readings differ from the {here} value by more than "
          f"5 percent; the widest gap for each signal:")
    for name in LIMIT_SIGNALS:
        q = widest.get(name)
        if q:
            print(f"    {name:>20}  {q['trial']:>9}  quoted {q['quoted']:g}, {here} {q['here']:g}")
    if key_agrees != len(first) or moved:
        raise SystemExit(f"the first-view points do not pair with the record (answer key "
                         f"agrees on {key_agrees} of {len(first)}, quoted readings move the "
                         f"rule on {', '.join(moved) or 'no point'}); not comparing")

    print(f"\n{'points':>12} {'n':>4} {'rule':>6} {'nano':>6} {'rule only':>10} "
          f"{'nano only':>10} {'exact McNemar p':>16}")
    print("-" * 70)
    for name, t in sets.items():
        print(f"{name:>12} {t['points']:>4} {t['rule_right']:>6} {t['nano_right']:>6} "
              f"{t['rule_only']:>10} {t['nano_only']:>10} {t['mcnemar_p']:>16}")

    print("\nwhere the rule and the smallest model differ:")
    for p in scored:
        if p["rule_look_again"] != p["nano_look_again"]:
            right = "rule" if p["rule_look_again"] == p["should_look_again"] else "nano"
            print(f"  {p['trial']:>9}  truth {'look' if p['should_look_again'] else 'accept':<6} "
                  f"rule {'look' if p['rule_look_again'] else 'accept':<6} "
                  f"nano {'look' if p['nano_look_again'] else 'accept':<6} "
                  f"gain {p['gain_mm']:+.3f} mm  {right} right  "
                  f"limits missed: {p['missed']}")
            # Read the reason before crediting the call: a right answer given
            # for a reason the evidence contradicts is not judgement.
            print(f"             nano said: {p['nano_reason'][:150]}")

    literal = sum(1 for p in scored if p["literal_look_again"])
    print(f"\nrule_decision called on the evidence exactly as the cell passes it (signals nested "
          f"under 'signals') asks for another look on {literal} of {len(scored)} points.")
    print("That is the fallback plan_next_view uses when a reply cannot be parsed. Until it was")
    print("taught the nested shape it found no signals and asked for another look on every point.")

    if stored is None:
        out = Path(args.rule_out)
        out.write_text(json.dumps({
            "command": "python scripts/eval_router_truth.py --rule",
            "answers": str(record).replace("\\", "/"),
            "first_view_source": str(manifest).replace("\\", "/"),
            "recorded_right": recorded,
            "answer_key_agrees": [key_agrees, len(first)],
            # Limits a reason compares against ("exceeds 0.6") are not counted.
            "quoted_readings": {
                "checked_one_at_a_time": len(quoted),
                "distinct": len(distinct),
                "points_where_the_rule_moves": moved,
                "distinct_differing_by_over_5_percent": len(differ),
                "widest_gap": {name: {"trial": q["trial"], "quoted": q["quoted"],
                                      "rebuilt": q["here"]}
                               for name, q in widest.items()},
            },
            "literal_fallback_look_again": [literal, len(scored)],
            "compare": sets,
            # The first-view signals, so the rule's count can be checked
            # without the render it was measured on.
            "first_view_points": [{
                "trial": p["trial"],
                "view_deg": p["evidence"][0]["view_deg"],
                "remaining": p["remaining"],
                "failed": bool(p["evidence"][0].get("measurement_failed")),
                "signals": {k: p["evidence"][0]["signals"][k] for k in LIMIT_SIGNALS}
                if "signals" in p["evidence"][0] else None,
                "should_look_again_rebuilt": p["should_look_again"],
            } for p in first],
        }, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="data/campaign50.jsonl")
    parser.add_argument("--manifest", default="data/cap50v3/manifest.json",
                        help="Captures measured from every angle for the first-view points.")
    parser.add_argument("--points", choices=("replayed", "first-view", "both"), default="both")
    parser.add_argument("--per-side", type=int, default=30,
                        help="First-view points kept per answer, look and accept, so the test "
                             "can catch both kinds of mistake.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the decision points and report them; call no model.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="data/router_truth_eval.jsonl")
    parser.add_argument("--rule", action="store_true",
                        help="Score the cell's arithmetic rule against the recorded answers of "
                             "the smallest model on the same points. Calls no model.")
    parser.add_argument("--record", default="data/router_truth_eval.jsonl",
                        help="The recorded model answers the rule arm is paired with.")
    parser.add_argument("--rule-out", default=str(RULE_RECORD),
                        help="Where the rule arm writes its counts and the first-view signals.")
    args = parser.parse_args()

    if args.rule:
        return run_rule_arm(args)

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

    arms = {tier_of(m): m for m in SIZES}
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
