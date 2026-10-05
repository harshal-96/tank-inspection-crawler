"""What the tank numbers can and cannot support.

    python tank/significance.py          # the reading, in sentences
    python tank/significance.py --json   # every number below, as one JSON object

Every headline in this project is a count out of a small number, and a count
out of a small number will happily look like a result when it is noise. This
works out what each one actually licenses, so the README can claim that and
not more. The --json output is the sheet the README's numbers are read from.

Nothing here calls a model. It reads the recorded runs and does arithmetic,
and every count is computed from them: the inspection rows go through the same
replay and score() as tank/score_survey.py, and the probe runs are read from
the files the comparison was recorded in.

Three questions.

  How good could it really be?   A count out of 24 is consistent with a system
                                 a good deal worse than the count, and the
                                 exact lower bound says how much worse. The
                                 frames overlap and hold five faults between
                                 them, so even that bound is generous. 13 of
                                 the 24 frames are clean, so the walk-in call
                                 is also split by class.
  Is the planner better?         Every recorded Nemotron run on the 28 point
                                 scene against the arithmetic sweep, paired, so
                                 exact McNemar, one p per run and never pooled,
                                 with the chance band and the patches reached
                                 beside it; then against the sweep shown only
                                 the model's view, and against the spread of
                                 the sweep's tie-break orders. The retired four
                                 point runs are reported apart, never pooled,
                                 to show why that design could not decide.
  Would it finish?               A real shell is 473 m of seam. At the measured
                                 rate per frame, serial inference is slower
                                 than the robot, so the hours are worked out
                                 for two ways of running it.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tank.score_probes import (  # noqa: E402
    RECORDED_RUNS,
    chance_distribution,
    fell_back_stations,
    model_only_hits,
    p_at_least,
    run,
    sweep_models_view,
    tiebreak_study,
)
from tank.score_survey import per_fault, replayed_rows, score  # noqa: E402
from tank.truth import (  # noqa: E402
    RADIUS,
    TRACE,
    in_frame,
    load_answers,
    project,
    wall_point,
)

SURVEY = ROOT / "tank" / "data" / "survey"
SCENE = ROOT / "tank" / "data" / "tank_scene.json"

# The recorded comparisons of the model planner with the sweep. Each holds the
# sweep and one or more Nemotron runs over the same stations. One list, kept in
# score_probes.py, so --tiebreaks and this script read the same runs.
PROBE_RUNS = RECORDED_RUNS

# The retired scene with 4 rejectable points. Read only by four_point(), which
# reports it apart: its runs are a different experiment and are never pooled
# with the 28 point ones or quoted as a current result.
ARCHIVE = SURVEY / "archive_four_point"

# An assumed crawling speed for a magnetic crawler carrying a spot probe. Not
# measured anywhere in this project; the hours below scale with it.
DRIVE_M_S = 0.10


def binomial_at_least(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson_lower(right: int, n: int, confidence: float = 0.95) -> float:
    """Exact one sided lower bound on the true rate behind `right` out of `n`.

    The lowest rate that would still produce at least this many successes
    with probability at least 1 - confidence (one run in twenty at the default
    0.95), found by bisection on the binomial tail. For a clean
    sweep it is alpha to the power one over n, which the bisection reproduces;
    no approximation is used anywhere, because on counts this small the usual
    ones are visibly wrong.
    """
    if n <= 0:
        raise ValueError("a bound on nothing: no trials")
    if not 0 <= right <= n:
        raise ValueError(f"{right} successes out of {n}")
    if right == 0:
        return 0.0
    alpha = 1.0 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binomial_at_least(right, n, mid) < alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two sided exact McNemar on the discordant pairs.

    Paired, because both planners are scored on the same probe points. The
    pairs where they agree carry no information about which is better, so
    the test runs on the points one planner found and the other did not.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    smaller = min(only_a, only_b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def smallest_attainable_p(points: int) -> float:
    """The lowest two sided exact McNemar p that `points` rejectable points allow.

    Two planners can disagree on at most every point, and the most extreme
    result is one planner alone on all of them. If even that cannot reach 0.05
    the design could not have shown a difference whatever the planners did.
    """
    return min(mcnemar_exact(n, 0) for n in range(1, points + 1))


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} is missing, and every number here is read from it")
    return json.loads(path.read_text(encoding="utf-8"))


def frame_extent_m(frame: dict) -> tuple[float, float]:
    """How much plate one frame shows, along the seam and across it, in metres.

    Measured on the plate surface through truth.project, the camera the answer
    key uses, in 1 mm steps.
    """
    theta, z = frame["theta_rad"], frame["z_m"]
    along = [s / 1000 for s in range(-1000, 1001)
             if in_frame(project(wall_point(theta + s / 1000 / RADIUS, z), theta, z))]
    if not along:
        raise SystemExit(f"station {frame['station']}: the frame shows no plate")
    # The lens sits ahead of the crawler's origin, so the middle of the
    # picture is measured rather than assumed.
    middle = theta + (max(along) + min(along)) / 2 / RADIUS
    across = [s / 1000 for s in range(-500, 501)
              if in_frame(project(wall_point(middle, z + s / 1000), theta, z))]
    return max(along) - min(along), max(across) - min(across)


# The sliver threshold the key uses, and the values it is tried at. Chosen to
# span "a few pixels" to "half the fault", not tuned to the result.
THRESHOLDS = (0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.25, 0.3, 0.5)


def threshold_sensitivity(frames: list[dict], answers: dict) -> list[dict]:
    """The same verdicts re-marked with the sliver threshold moved.

    The key counts a fault as in the picture once it fills TRACE of itself or
    of the frame. That number is a choice, so the scores it produces are
    printed across a range of it rather than at the one value alone.
    """
    from tank import truth

    kept = truth.TRACE
    out = []
    try:
        for threshold in THRESHOLDS:
            truth.TRACE = threshold
            got = score(replayed_rows(frames, answers))
            out.append({"threshold": threshold,
                        "pass_or_defect": got["pass_or_defect_right"],
                        "right_call_for_a_fault_in_view": got["gate_right_for_a_fault_in_view"],
                        "faults_found": got["faults_found"],
                        "faults_present": got["faults_present"],
                        "headline_named": got["named_right"],
                        "missed": got["missed"],
                        "false_alarms": got["false_alarms"],
                        "judged": got["judged"]})
    finally:
        truth.TRACE = kept
    return out


def inspection(manifest: dict) -> dict:
    """The recorded verdicts, replayed and bounded."""
    path = SURVEY / "inspection.jsonl"
    models = {json.loads(line).get("model")
              for line in path.open(encoding="utf-8") if line.strip()}
    if len(models) != 1:
        raise SystemExit(f"{path} holds verdicts from {len(models)} models; "
                         "the bounds below are for one")
    frames = manifest["frames"]
    rows = replayed_rows(frames, load_answers(path))
    got = score(rows)
    if got["judged"] == 0:
        raise SystemExit("no judged frames to bound")

    def measure(right: int, n: int) -> dict:
        return {"right": right, "of": n,
                "lower_95_one_sided": round(clopper_pearson_lower(right, n), 4)}

    judged = got["judged"]
    measures = {
        "pass_or_defect": measure(got["pass_or_defect_right"], judged),
        "right_call_for_a_fault_in_view": measure(got["gate_right_for_a_fault_in_view"],
                                                  judged),
        "faults_found": measure(got["faults_found"], got["faults_present"]),
        "headline_named": measure(got["named_right"], judged),
        "frames_with_every_fault_found": measure(got["frames_all_found"], judged),
    }
    # The walk-in call split by what the frame holds. Most of the pooled count
    # is clean frames passed, which calling every frame sound would also get,
    # so the frames that hold a fault are bounded on their own.
    judged_rows = [r for r in rows if r["verdict"] != "uncertain"]
    faulty = [r for r in judged_rows if r["truth"] != "none"]
    clean = [r for r in judged_rows if r["truth"] == "none"]
    if not faulty or not clean:
        raise SystemExit("the run needs both clean frames and frames holding a fault")
    per_class = {
        "frames_holding_a_fault_flagged": measure(
            sum(1 for r in faulty if r["called"] != "none"), len(faulty)),
        "frames_holding_a_fault_flagged_for_a_fault_in_view": measure(
            sum(1 for r in faulty if r["called"] != "none" and r["called"] in r["in_view"]),
            len(faulty)),
        "clean_frames_passed": measure(
            sum(1 for r in clean if r["called"] == "none"), len(clean)),
        "calling_every_frame_sound_scores": len(clean),
    }
    faults = per_fault(frames, rows)
    named = sum(1 for f in faults if f["named_in"])
    if not faults:
        raise SystemExit("no faults on the run to count")
    along, across = frame_extent_m(frames[len(frames) // 2])
    step = manifest["step_m"]
    return {
        "model": models.pop(),
        "frames": got["frames"], "judged": judged, "declined": got["declined"],
        "measures": measures,
        "per_class": per_class,
        "threshold_in_use": TRACE,
        "threshold_sensitivity": threshold_sensitivity(frames, load_answers(path)),
        "invented": got["invented"], "false_alarms": got["false_alarms"],
        "missed": got["missed"],
        "missed_stations": [r["station"] for r in rows if r["verdict"] != "uncertain"
                            and r["truth"] != "none" and r["called"] == "none"],
        "called_a_sliver": got["called_a_sliver"],
        "faults_on_the_run": {
            **measure(named, len(faults)),
            "never_named": [{"id": f["id"], "kind": f["kind"], "frames": f["frames"]}
                            for f in faults if not f["named_in"]],
            "sightings": sum(len(f["frames"]) for f in faults)},
        "overlap": {"step_m": step,
                    "frame_along_seam_m": round(along, 3),
                    "frame_across_seam_m": round(across, 3),
                    "share_of_a_frame_also_in_the_next": round(max(0.0, 1 - step / along), 3)},
    }


def probing(manifest: dict) -> dict:
    """Every recorded Nemotron run on this scene, against the sweep."""
    frames = manifest["frames"]
    where = {(f["station"], round(c["offset_m"], 4)): c
             for f in frames for c in f["probe_candidates"]}
    rejectable = sum(1 for c in where.values() if c["rejectable"])
    hidden = sorted({i for c in where.values() if c["rejectable"] for i in c["thinned_by"]})
    budget = None

    def patches(found) -> list[str]:
        return sorted({i for s, o, *_ in found
                       for i in where[(s, round(o, 4))]["thinned_by"]})

    runs, skipped, rule = [], [], None
    for name in PROBE_RUNS:
        data = load(SURVEY / name)
        # The retired scene had 4 rejectable points, not 28. A run scored on it
        # is a different experiment and is not pooled with these.
        if data["rejectable"] != rejectable:
            skipped.append({"file": name, "rejectable": data["rejectable"]})
            continue
        if budget is None:
            budget = data["budget"]
        elif data["budget"] != budget:
            raise SystemExit(f"{name} was run with a budget of {data['budget']}, "
                             f"not {budget}")
        sweep = next((p for p in data["planners"] if p["label"].startswith("sweep")), None)
        if sweep is None:
            raise SystemExit(f"{name} has no sweep to compare against")
        rule_hits = {(s, round(o, 4)) for s, o, *_ in sweep["rejectable_found"]}
        if rule is None:
            rule = {"label": sweep["label"], "found": len(rule_hits),
                    "spent": sweep["spent"], "hits": rule_hits,
                    "patches_found": patches(sweep["rejectable_found"])}
        elif rule_hits != rule["hits"]:
            raise SystemExit(f"the sweep in {name} found different points from the "
                             "sweep in the first file; it is meant to be the same rule")
        for p in data["planners"]:
            if "Nemotron" not in p["label"]:
                continue
            hits = {(s, round(o, 4)) for s, o, *_ in p["rejectable_found"]}
            only_model, only_rule = len(hits - rule_hits), len(rule_hits - hits)
            got = patches(p["rejectable_found"])
            # A station whose reply was unreadable was planned by the sweep, so
            # its finds are the rule's. The count the model earned itself
            # leaves them out; padding cannot be left out of the old runs,
            # which did not record it (see score_probes.fell_back_stations).
            # The test drops those stations from both arms: the model planned
            # nothing there, so the rule's finds there are not a model miss.
            fell = set(fell_back_stations(p) or [])
            own = {(s, round(o, 4)) for s, o, *_ in model_only_hits(p)}
            rule_own = {h for h in rule_hits if int(h[0]) not in fell}
            runs.append({"file": name, "label": p["label"], "attempt": p.get("attempt"),
                         "spent": p["spent"], "found": len(hits),
                         "patches_found": got,
                         "patches_missed": [i for i in hidden if i not in got],
                         "only_model": only_model, "only_rule": only_rule,
                         "mcnemar_p": round(mcnemar_exact(only_model, only_rule), 4),
                         "fell_back_stations": fell_back_stations(p),
                         "padding_recorded": "padded" in p,
                         "model_only_found": len(own),
                         "model_only_mcnemar_p": round(mcnemar_exact(
                             len(own - rule_own), len(rule_own - own)), 4),
                         "hits": hits,
                         "seconds": p.get("seconds")})
    if rule is None or not runs:
        raise SystemExit("no recorded Nemotron run on this scene to compare")
    # The same runs against the sweep held to the model's view, at its shipped
    # tie-break order. Arithmetic, so it is recomputed rather than read; the
    # recorded baselines file holds the same arm.
    viewed = run(frames, {}, lambda o, s, m, b, f, r: sweep_models_view(o, s, m, b),
                 budget, "sweep, model's view")
    view_hits = {(s, round(o, 4)) for s, o, *_ in viewed["rejectable_found"]}
    for r in runs:
        hits = r.pop("hits")
        r["only_model_vs_view"] = len(hits - view_hits)
        r["only_view"] = len(view_hits - hits)
        r["mcnemar_p_vs_view"] = round(mcnemar_exact(len(hits - view_hits),
                                                     len(view_hits - hits)), 4)
    # The sweep's 16 is one tie-break order of 5040. Both arms under all of
    # them, so a run is read against the rule's spread and not that one order.
    study = tiebreak_study(frames, {}, budget, rejectable)
    paired = study["full_minus_models_view"]
    tiebreaks = {"orders": study["orders"],
                 "full_median": study["median"], "full_mean": round(study["mean"], 2),
                 "full_range": [study["min"], study["max"]],
                 "view_median": study["models_view"]["median"],
                 "view_mean": round(study["models_view"]["mean"], 2),
                 "view_range": [study["models_view"]["min"], study["models_view"]["max"]],
                 "full_minus_view_median": paired["median"],
                 "full_minus_view_mean": round(paired["mean"], 2),
                 "share_full_ahead": round(paired["share_full_ahead"], 4),
                 "share_level": round(paired["share_level"], 4),
                 "share_full_behind": round(paired["share_full_behind"], 4)}
    chance = chance_distribution(frames, budget)
    for r in runs:
        r["chance_at_least"] = round(p_at_least(chance, r["found"]), 4)
    rule["chance_at_least"] = round(p_at_least(chance, rule["found"]), 4)
    del rule["hits"]
    ps = [r["mcnemar_p"] for r in runs]
    return {"rejectable": rejectable, "places": len(where), "budget": budget,
            "hidden_patches": hidden,
            "chance": {"mean": round(chance["mean"], 4), "band_95": chance["band_95"],
                       "band_mass": round(chance["band_mass"], 4)},
            "rule": rule, "runs": runs, "skipped": skipped,
            "models_view": {"label": viewed["label"],
                            "found": len(viewed["rejectable_found"]),
                            "patches_found": viewed["patches_found"]},
            "tiebreaks": tiebreaks,
            "mcnemar_p_range": [min(ps), max(ps)],
            "model_found_range": [min(r["found"] for r in runs),
                                  max(r["found"] for r in runs)],
            "rule_ahead_in_every_run": all(r["found"] < rule["found"] for r in runs),
            "model_ahead_in_any_run": any(r["found"] > rule["found"] for r in runs)}


def four_point() -> dict:
    """The retired four point runs, each paired against its own sweep.

    Reported because a public figure was once read off them; kept apart from
    probing() because they were scored on another scene. The point they make
    is about the design: with 4 rejectable points the smallest two sided exact
    p there is, one planner alone on all four, is 0.125, so no run on that
    scene could have shown a difference at 0.05.
    """
    runs, rejectable = [], None
    for path in sorted(ARCHIVE.glob("*.json")):
        data = load(path)
        if rejectable is None:
            rejectable = data["rejectable"]
        elif data["rejectable"] != rejectable:
            raise SystemExit(f"{path.name} holds {data['rejectable']} rejectable points, "
                             f"the other archived runs {rejectable}")
        sweep = next((p for p in data["planners"] if p["label"].startswith("sweep")), None)
        if sweep is None:
            raise SystemExit(f"{path.name} has no sweep to compare against")
        rule_hits = {(s, round(o, 4)) for s, o, *_ in sweep["rejectable_found"]}
        for p in data["planners"]:
            if "Nemotron" not in p["label"]:
                continue
            hits = {(s, round(o, 4)) for s, o, *_ in p["rejectable_found"]}
            runs.append({"file": path.name, "label": p["label"],
                         "attempt": p.get("attempt"), "found": len(hits),
                         "rule_found": len(rule_hits),
                         "only_model": len(hits - rule_hits),
                         "only_rule": len(rule_hits - hits),
                         "mcnemar_p": round(mcnemar_exact(len(hits - rule_hits),
                                                          len(rule_hits - hits)), 4)})
    if not runs:
        raise SystemExit(f"no archived runs under {ARCHIVE}")
    return {"rejectable": rejectable, "runs": runs,
            "smallest_attainable_p": smallest_attainable_p(rejectable)}


def hours(manifest: dict, probe: dict) -> dict:
    """What one tank would take, two ways of running the inference."""
    scene = load(SCENE)
    seam_m = sum(s["length_m"] for s in scene["seams"])
    step = manifest["step_m"]
    stations = seam_m / step
    latencies = [json.loads(line)["latency_ms"]
                 for line in (SURVEY / "inspection.jsonl").open(encoding="utf-8")
                 if line.strip()]
    if not latencies:
        raise SystemExit("no recorded latencies")
    # The hours use the median frame, so one slow request does not set the
    # rate for 2720 stations. A serial total is really stations times the
    # mean, and the frame times are skewed upward, so the serial hours are
    # given at the mean as well and the median figure is the kinder of the two.
    look_s = statistics.median(latencies) / 1000
    mean_s = statistics.fmean(latencies) / 1000
    station_drive_s = step / DRIVE_M_S
    timed = [r["seconds"] / len(manifest["frames"]) for r in probe["runs"] if r["seconds"]]
    plan_s = statistics.median(timed) if timed else None
    concurrent = math.ceil(look_s / station_drive_s)
    drive_h = seam_m / DRIVE_M_S / 3600
    look_h = stations * look_s / 3600
    out = {"seam_m": round(seam_m, 1), "step_m": step, "stations": round(stations),
           "drive_m_s_assumed": DRIVE_M_S,
           "median_frame_s": round(look_s, 2),
           "mean_frame_s": round(mean_s, 2),
           "slowest_frame_s": round(max(latencies) / 1000, 2),
           "hours_use": "median",
           "drive_s_a_station": round(station_drive_s, 2),
           "drive_h": round(drive_h, 2),
           "serial_vision_h": round(look_h, 2),
           "serial_vision_h_at_mean": round(stations * mean_s / 3600, 2),
           "concurrent_requests_at_mean": math.ceil(mean_s / station_drive_s),
           "stop_and_look_h": round(drive_h + look_h, 2),
           "concurrent_requests_to_keep_up": concurrent,
           "look_while_driving_h": round(max(drive_h, look_h / concurrent), 2)}
    if plan_s is not None:
        out["median_plan_s"] = round(plan_s, 2)
        out["model_planner_serial_h"] = round(stations * plan_s / 3600, 2)
        # Each plan needs the readings the station before returned, so the
        # planner cannot be spread over concurrent requests. At best it runs
        # while the crawler drives to the next station.
        out["model_planner_overlapped_h"] = round(
            stations * max(0.0, plan_s - station_drive_s) / 3600, 2)
    return out


def report(numbers: dict) -> None:
    ins, probe, hrs = numbers["inspection"], numbers["probing"], numbers["hours"]
    m = ins["measures"]

    def line(label, key):
        x = m[key]
        print(f"  {label:<46} {x['right']:>3} of {x['of']:<3} "
              f"lower bound {x['lower_95_one_sided']:.3f}")

    print("WHAT THE INSPECTION NUMBERS LICENSE\n")
    print(f"  {ins['model']}, {ins['frames']} frames, {ins['judged']} judged, "
          f"{ins['declined']} declined. One sided 95 percent exact lower bounds.\n")
    line("pass or defect right", "pass_or_defect")
    line("right call for a fault actually in view", "right_call_for_a_fault_in_view")
    line("faults found, all faults in each frame", "faults_found")
    line("headline (worst fault) named", "headline_named")
    line("frames with every fault found", "frames_with_every_fault_found")
    f = ins["faults_on_the_run"]
    print(f"  {'distinct faults named in a frame holding them':<46} {f['right']:>3} of "
          f"{f['of']:<3} lower bound {f['lower_95_one_sided']:.3f}")
    for miss in f["never_named"]:
        print(f"    never named: {miss['id']} {miss['kind']}, in frames "
              f"{', '.join(map(str, miss['frames']))}")
    print(f"\n  missed (called none with a fault in view): {ins['missed']}, at station(s) "
          f"{', '.join(map(str, ins['missed_stations'])) or '-'}. "
          f"False alarms {ins['false_alarms']}, invented {ins['invented']}.")
    c = ins["per_class"]
    print(f"\n  The pass or defect call by class. Calling every frame sound scores "
          f"{c['calling_every_frame_sound_scores']} of {ins['judged']},\n  so the pooled "
          f"count is mostly clean frames; the frames holding a fault are the test:")
    for label, key in (("frames holding a fault, flagged", "frames_holding_a_fault_flagged"),
                       ("  flagged for a fault really in view",
                        "frames_holding_a_fault_flagged_for_a_fault_in_view"),
                       ("clean frames passed", "clean_frames_passed")):
        x = c[key]
        print(f"  {label:<46} {x['right']:>3} of {x['of']:<3} "
              f"lower bound {x['lower_95_one_sided']:.3f}")
    o = ins["overlap"]
    print(f"\n  These are not independent trials. A frame shows {o['frame_along_seam_m']:.3f} m "
          f"of seam and the\n  stations are {o['step_m']} m apart, so "
          f"{o['share_of_a_frame_also_in_the_next'] * 100:.0f} percent of each frame is in "
          f"the next one too,\n  and the {f['sightings']} sightings of a fault are "
          f"{f['of']} physical faults. The bounds treat\n  each frame as a fresh draw, "
          f"so they are, if anything, too kind.\n")

    print(f"  The same verdicts with the sliver threshold moved (the key uses "
          f"{ins['threshold_in_use']}):")
    print(f"  {'threshold':>10} {'pass or defect':>15} {'fault in view':>14} "
          f"{'faults found':>13} {'headline':>9} {'missed':>7}")
    for t in ins["threshold_sensitivity"]:
        print(f"  {t['threshold']:>10} {t['pass_or_defect']:>9} of {t['judged']:<3} "
              f"{t['right_call_for_a_fault_in_view']:>11}    "
              f"{t['faults_found']:>6} of {t['faults_present']:<4} {t['headline_named']:>7} "
              f"{t['missed']:>7}")
    print()

    print("IS THE PROBE PLANNER ACTUALLY BETTER\n")
    rule, chance = probe["rule"], probe["chance"]
    print(f"  {probe['rejectable']} rejectable of {probe['places']} places, "
          f"{probe['budget']} probes a station, on {len(probe['hidden_patches'])} "
          f"hidden patches: {', '.join(probe['hidden_patches'])}")
    print(f"  picking {probe['budget']} at random: {chance['mean']:.1f} expected, "
          f"central 95 percent {chance['band_95'][0]} to {chance['band_95'][1]} "
          f"(holds {chance['band_mass'] * 100:.1f} percent; tank/score_probes.py)")
    print(f"  {rule['label']}: {rule['found']} found, "
          f"{len(rule['patches_found'])} of {len(probe['hidden_patches'])} patches, "
          f"chance of a random planner doing as well {rule['chance_at_least']:.4f}\n")
    print(f"  {'run':<50} {'found':>6} {'patches':>8} {'model only':>11} "
          f"{'rule only':>10} {'p':>7} {'chance >=':>10}")
    for r in probe["runs"]:
        name = f"{r['file']}: {r['label']}" + (f" #{r['attempt']}" if r["attempt"] else "")
        print(f"  {name:<50} {r['found']:>6} "
              f"{len(r['patches_found']):>4}/{len(probe['hidden_patches']):<3} "
              f"{r['only_model']:>11} {r['only_rule']:>10} {r['mcnemar_p']:>7.3f} "
              f"{r['chance_at_least']:>10.4f}"
              + (f"   missed {', '.join(r['patches_missed'])}" if r["patches_missed"] else ""))
    for s in probe["skipped"]:
        print(f"  skipped {s['file']}: scored on another scene "
              f"({s['rejectable']} rejectable)")
    for r in probe["runs"]:
        name = f"{r['file']} #{r['attempt']}" if r["attempt"] else r["file"]
        if r["fell_back_stations"] is None:
            print(f"  {name}: no reasons recorded, so its fallbacks cannot be counted")
        elif r["fell_back_stations"]:
            stations = ", ".join(map(str, r["fell_back_stations"]))
            print(f"  {name}: the reply at station(s) {stations} was unreadable and the "
                  f"rule chose there,\n    so the model's own picks found "
                  f"{r['model_only_found']}, not {r['found']}; excluding station(s) "
                  f"{stations} from both arms,\n    p {r['model_only_mcnemar_p']:.3f}")
    if not all(r["padding_recorded"] for r in probe["runs"]):
        print("  Replies that named fewer offsets than the budget were topped up from "
              "the rule, and\n  these runs did not record where; live runs record it "
              "from now on.")
    lo, hi = probe["mcnemar_p_range"]
    n = len(probe["runs"])
    print(f"\n  exact McNemar p per run, not pooled: {lo:.3f} to {hi:.3f}")
    # Read off the data rather than asserted, because this conclusion has
    # already flipped once: on four rejectable points the model looked better.
    significant = sum(1 for r in probe["runs"] if r["mcnemar_p"] < 0.05)
    if probe["rule_ahead_in_every_run"]:
        print(f"  At its shipped tie-break order the rule found more than the model in "
              f"all {n} recorded runs;\n  {significant} of {n} is significant at 0.05 "
              f"on its own.")
        view_found = probe["models_view"]["found"]
        level = [r for r in probe["runs"] if r["found"] >= view_found]
        if level:
            print("  Against the sweep shown only the model's view it did not hold:\n    "
                  + ";\n    ".join(f"{r['file']}" + (f" #{r['attempt']}" if r["attempt"] else "")
                              + f" found {r['found']} to its {view_found}" for r in level)
                  + ".")
    elif probe["model_ahead_in_any_run"]:
        print("  The lead changed hands between runs. Nothing to choose between them.")
    else:
        print("  The model never beat the rule, and tied it at least once.")
    print(f"  The points cluster in {len(probe['hidden_patches'])} patches, so the "
          f"pairs are not independent either.\n")

    # The 16 is one tie-break order. Read against the rule's spread, and
    # against the rule given only what the model is shown, the gap shrinks.
    view, tb = probe["models_view"], probe["tiebreaks"]
    print(f"  That rule is one tie-break order of {tb['orders']}. Over all of them the "
          f"sweep finds\n  {tb['full_range'][0]} to {tb['full_range'][1]}, median "
          f"{tb['full_median']:g} (mean {tb['full_mean']:.2f}); shown only the model's view, "
          f"{tb['view_range'][0]} to {tb['view_range'][1]},\n  median {tb['view_median']:g} "
          f"(mean {tb['view_mean']:.2f}). Paired order by order, full history minus model's "
          f"view:\n  median {tb['full_minus_view_median']:+g}, mean "
          f"{tb['full_minus_view_mean']:+.2f}; full history ahead in "
          f"{tb['share_full_ahead'] * 100:.1f}% of orders, level in "
          f"{tb['share_level'] * 100:.1f}%, behind in {tb['share_full_behind'] * 100:.1f}%.")
    print("  " + ("Withholding the history does not hurt the rule, so the state it "
                  "keeps is not what\n  separates it from the model."
                  if tb["full_minus_view_median"] <= 0 else
                  "Withholding the history lowers the rule's typical count."))
    print(f"\n  Against the sweep shown only the model's view, at its shipped order "
          f"({view['found']} found):")
    print(f"  {'run':<50} {'found':>6} {'model only':>11} {'view only':>10} {'p':>7}")
    for r in probe["runs"]:
        name = f"{r['file']}: {r['label']}" + (f" #{r['attempt']}" if r["attempt"] else "")
        print(f"  {name:<50} {r['found']:>6} {r['only_model_vs_view']:>11} "
              f"{r['only_view']:>10} {r['mcnemar_p_vs_view']:>7.3f}")
    print()

    fp = numbers["four_point"]
    print(f"  The retired four point scene ({fp['rejectable']} rejectable), each run "
          f"against its own sweep,\n  reported apart and never pooled with the runs above:")
    print(f"  {'run':<56} {'found':>6} {'rule':>5} {'model only':>11} "
          f"{'rule only':>10} {'p':>7}")
    for r in fp["runs"]:
        name = f"{r['file']}: {r['label']}" + (f" #{r['attempt']}" if r["attempt"] else "")
        print(f"  {name:<56} {r['found']:>6} {r['rule_found']:>5} {r['only_model']:>11} "
              f"{r['only_rule']:>10} {r['mcnemar_p']:>7.3f}")
    print(f"  On {fp['rejectable']} points the smallest two sided exact p attainable is "
          f"{fp['smallest_attainable_p']:.3f}, one planner\n  alone on every point, so that "
          f"design could not show a difference at 0.05 whatever happened.\n")

    print("WOULD IT FINISH A REAL TANK\n")
    print(f"  {hrs['seam_m']:.0f} m of seam, a station every {hrs['step_m'] * 1000:.0f} mm "
          f"= {hrs['stations']} stations")
    print(f"  driving at {hrs['drive_m_s_assumed']:.2f} m/s (assumed):             "
          f"{hrs['drive_h']:5.1f} hours")
    print(f"  vision at a median {hrs['median_frame_s']:.1f} s a frame, serial:  "
          f"{hrs['serial_vision_h']:5.1f} hours")
    print(f"  (frame time median {hrs['median_frame_s']:.2f} s, mean "
          f"{hrs['mean_frame_s']:.2f} s, slowest {hrs['slowest_frame_s']:.2f} s; the hours "
          f"here use the {hrs['hours_use']}.\n  At the mean, serial vision is "
          f"{hrs['serial_vision_h_at_mean']:.1f} hours and "
          f"{hrs['concurrent_requests_at_mean']} requests in flight still keep up.)")
    print(f"\n  stop and look at every station:     {hrs['stop_and_look_h']:5.1f} hours")
    print(f"  look while driving, {hrs['concurrent_requests_to_keep_up']} requests in "
          f"flight: {hrs['look_while_driving_h']:5.1f} hours")
    print(f"  ({hrs['drive_s_a_station']:.2f} s of driving a station against "
          f"{hrs['median_frame_s']:.1f} s a frame,\n  so {hrs['concurrent_requests_to_keep_up']} "
          f"concurrent requests keep the inference at drive speed.)")
    if "median_plan_s" in hrs:
        # Each run's seconds over its stations, then the median of those runs.
        print(f"\n  a model planner at a median of {hrs['median_plan_s']:.1f} s a station adds "
              f"{hrs['model_planner_serial_h']:.1f} hours run serially, and still "
              f"{hrs['model_planner_overlapped_h']:.1f}\n  overlapped with driving: each "
              f"plan needs the last station's readings, so it\n  cannot be spread over "
              f"concurrent requests. The arithmetic rule costs nothing.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true",
                        help="print every number as one JSON object instead of prose")
    args = parser.parse_args()

    manifest = load(SURVEY / "manifest.json")
    probe = probing(manifest)
    numbers = {"inspection": inspection(manifest), "probing": probe,
               "four_point": four_point(), "hours": hours(manifest, probe)}
    if args.json:
        print(json.dumps(numbers, indent=1))
    else:
        report(numbers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
