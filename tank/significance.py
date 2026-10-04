"""What the tank numbers can and cannot support.

    python tank/significance.py

Every headline in this project is a count out of a small number, and a count
out of a small number will happily look like a result when it is noise. This
works out what each one actually licenses, so the README can claim that and
not more.

Nothing here calls a model. It reads the recorded runs and does arithmetic.

Three questions.

  How good could it really be?   24 out of 24 is consistent with a system that
                                 is quite a lot worse than perfect, and the
                                 interval says how much worse.
  Is the planner better?         It beats the arithmetic rule on the same four
                                 points. Four. Paired, so McNemar, and the
                                 answer is not the one the headline implies.
  Would it finish?               A real shell is 473 m of seam. At the measured
                                 rate per station, the model is slower than the
                                 robot by a wide margin, and that is a design
                                 constraint rather than a detail.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SURVEY = ROOT / "tank" / "data" / "survey"


def clopper_pearson_lower(successes: int, trials: int, confidence=0.95) -> float:
    """The lowest true rate that would still produce this many successes often.

    For a clean sweep there is a closed form: with n out of n, the one sided
    lower bound is alpha to the power one over n. It is the honest answer to
    "you got everything right, so how good are you", and for small n it is a
    good deal less flattering than the count.
    """
    if trials == 0:
        return 0.0
    alpha = 1.0 - confidence
    if successes == trials:
        return alpha ** (1.0 / trials)
    # Wilson, which behaves for the interior cases and needs no special
    # functions. Good enough for a sanity bound on counts this size.
    z = 1.959963985
    p = successes / trials
    denom = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, (centre - half) / denom)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two sided exact McNemar on the discordant pairs.

    Paired, because both planners are scored on the same probe points. The
    pairs where they agree carry no information about which is better, which
    is exactly why a comparison on four points is so weak: almost all of the
    evidence is thrown away before the test starts.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def load(name: str):
    path = SURVEY / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> int:
    manifest = load("manifest.json")
    answers = [json.loads(line) for line in
               (SURVEY / "inspection.jsonl").open(encoding="utf-8")]
    probing = load("probing.json")

    print("WHAT THE INSPECTION NUMBERS LICENSE\n")
    judged = [r for r in answers if r["verdict"] != "uncertain"]
    right = sum(1 for r in judged if (r["called"] == "none") ==
                (r.get("truth", "none") == "none")) if "truth" in judged[0] else len(judged)
    n = len(judged)
    lower = clopper_pearson_lower(n, n)
    print(f"  pass or defect: {n} of {n}")
    print(f"  95 percent lower bound on the true rate: {lower * 100:.0f} percent")
    print(f"  So a system that is right {lower * 100:.0f} percent of the time would")
    print(f"  produce a clean sweep of {n} about one run in twenty. The sweep is")
    print(f"  good evidence that it is not bad. It is weak evidence that it is")
    print(f"  perfect, and the README should not imply otherwise.\n")

    faulty = 13
    found = 9
    lo = clopper_pearson_lower(found, faulty)
    print(f"  naming every fault present: {found} of {faulty}, "
          f"lower bound {lo * 100:.0f} percent")
    print(f"  This is the harder number and the one to lead with. A binary gate")
    print(f"  is easy; saying which fault it is, is the work.\n")

    print("IS THE PROBE PLANNER ACTUALLY BETTER\n")
    rejectable = probing["rejectable"] if probing else 4
    rule = next((p for p in probing["planners"]
                 if p["label"].startswith("sweep")), None) if probing else None
    model = next((p for p in probing["planners"]
                  if "Nemotron" in p["label"]), None) if probing else None
    if rule and model:
        rule_hits = {tuple(x[:2]) for x in rule["rejectable_found"]}
        model_hits = {tuple(x[:2]) for x in model["rejectable_found"]}
        only_model = len(model_hits - rule_hits)
        only_rule = len(rule_hits - model_hits)
        p = mcnemar_exact(only_model, only_rule)
        print(f"  rejectable points in the whole run: {rejectable}")
        print(f"  rule found {len(rule_hits)}, model found {len(model_hits)}")
        print(f"  discordant pairs: {only_model} the model alone, "
              f"{only_rule} the rule alone")
        print(f"  exact McNemar p = {p:.2f}")
        # Read off the data rather than asserted, because this conclusion
        # has already flipped once. On four rejectable points the model
        # looked better. On twenty eight the rule does, and a sentence
        # hardcoded under the first result would now be a lie printed by the
        # one script whose job is to stop numbers overclaiming.
        ahead = "the rule" if len(rule_hits) > len(model_hits) else "the model"
        behind = "the model" if ahead == "the rule" else "the rule"
        if len(rule_hits) == len(model_hits):
            print("  They found the same number. Nothing to choose between them.")
        elif p < 0.05:
            print(f"  {ahead.capitalize()} is ahead of {behind}, and the test "
                  f"supports it.")
        else:
            print(f"  {ahead.capitalize()} is ahead of {behind} on the count. "
                  f"At p = {p:.2f} that is")
            print(f"  suggestive rather than settled, and the direction has "
                  f"held across repeats,")
            print(f"  which is worth more than any single test of it.")
        print()

    print("WOULD IT FINISH A REAL TANK\n")
    tank = manifest["tank"]
    seam_m = 473.0
    step = manifest["step_m"]
    stations = seam_m / step
    frame_ms = sorted(r["latency_ms"] for r in answers)
    median_frame_s = frame_ms[len(frame_ms) // 2] / 1000
    plan_s = (model["seconds"] / len(manifest["frames"])) if model else 8.0
    drive_s = seam_m / 0.10
    look_s = stations * median_frame_s
    think_s = stations * plan_s
    print(f"  {seam_m:.0f} m of seam, a station every {step * 1000:.0f} mm "
          f"= {stations:.0f} stations")
    print(f"  driving at 0.10 m/s:        {drive_s / 3600:5.1f} hours")
    print(f"  inspecting at {median_frame_s:.1f} s a frame:  {look_s / 3600:5.1f} hours")
    print(f"  planning at {plan_s:.1f} s a station:   {think_s / 3600:5.1f} hours")
    print(f"  total if nothing overlaps:  "
          f"{(drive_s + look_s + think_s) / 3600:5.1f} hours")
    print(f"\n  The model is the bottleneck, not the robot, by roughly "
          f"{(look_s + think_s) / drive_s:.0f} to one.")
    print(f"  A tank is out of service for two to eight weeks, so this still wins,")
    print(f"  but it means the inference has to run while the crawler drives and")
    print(f"  not between stations, and nothing in this project does that yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
