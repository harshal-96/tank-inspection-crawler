"""Score the probe planner on the survey run, against answers written first.

    python tank/score_probes.py --baselines                # arithmetic only, writes nothing
    python tank/score_probes.py --baselines --out PATH     # the same, kept in a file
    python tank/score_probes.py --tiebreaks                # the sweep under every tie-break order
    python tank/score_probes.py --live --repeat 3          # the model arm: paid, into runs/

The run offers 168 places to measure and there is time for 48 of them. 28 of
the 168 fail the screening limit. They sit on five patches of wall loss with
nothing on the surface, and none of them is on the one patch the camera can
see, which measures inside limits. 20 of the 28 are under frames the inspector
passes. The script prints these counts from the manifest every time it runs,
so this paragraph can be checked against it.

So this measures one thing: not whether a model can see rust, but whether a
planner will spend a scarce measurement on plate that looks fine.

The planners compared on the same stations:

  on the seam    offset 0.0, then the nearest offsets to it. With a budget of
                 two its second pick is always -0.15, below the seam, and never
                 +0.15. What a fixed program does.
  probe beside reported rust
                 when the inspector names corrosion, the two offsets nearest
                 the seam on either side of it; on the seam otherwise. What
                 steering the probe by the camera alone can do on this rig.
  sweep          coverage plus return to thin readings: chase a low reading
                 from the last two stations, else the offset longest unmeasured.
                 Arithmetic, no model, and the thing to beat.
  sweep, model's view
                 the same rule given only the state the model's prompt carries,
                 so the comparison with the model is matched on information.
  Nemotron       the same job given to an NVIDIA model, one call a station.
                 Only with --live, because it is paid for.

Two references that are not planners and are kept apart from them: chance, the
exact distribution of what picking two of the seven at random finds, and the
oracle, the most any planner could find with two of seven at every station.

The verdicts come from tank/data/survey/inspection.jsonl, already paid for, so
nothing here calls the vision model again.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import nebius, registry  # noqa: E402
from tank.probe_planner import (  # noqa: E402
    DEFAULT_BUDGET,
    Memory,
    Reading,
    plan_probes,
    plan_with_floor,
    planner_model,
    sweep_rule,
)
from tank.truth import load_answers, truth_for  # noqa: E402

DATA = ROOT / "tank" / "data"
SURVEY = DATA / "survey"
RUNS = SURVEY / "runs"

# plan_probes shows the model memory.readings[-6:] and nothing older. The
# "model's view" arm is held to the same window, so if the prompt changes this
# has to change with it.
SHOWN_READINGS = 6


def said(row: dict) -> str:
    """The inspector's reading of the frame, in the words it used."""
    if row is None:
        return "no inspection of this frame"
    faults = row.get("faults") or ([] if row["called"] == "none" else [row["called"]])
    if not faults:
        return "sound seam, nothing on the surface"
    where = row.get("where") or "not stated"
    return f"{', '.join(faults)} ({where} of the frame)"


def named(row: dict | None) -> list[str]:
    """The faults the inspector named in a frame, as words."""
    if not row:
        return []
    return list(row.get("faults") or ([] if row["called"] == "none" else [row["called"]]))


def on_the_seam(offsets, station, memory, budget):
    """Offset 0.0, then the offsets nearest to it.

    The padding is a stable sort on distance from the seam, and the offsets
    arrive listed from -0.45 upward, so of each mirrored pair the one below the
    seam comes first. With a budget of two the second pick is therefore always
    -0.15 and never +0.15. Anything this arm finds off the seam it finds there.
    """
    picked = [o for o in offsets if abs(o) < 1e-6]
    return (picked + [o for o in sorted(offsets, key=abs) if o not in picked])[:budget]


def where_it_saw(offsets, station, memory, budget, row=None):
    """Probe beside reported rust, the most the camera can steer on this rig.

    The inspector reports where a fault is as left, middle or right of the
    frame. Those are directions along the seam: the camera's right axis is the
    seam's tangent and its up axis points down the wall (truth.camera_pose), and
    the probe offsets run up and down the wall. So the reported side says
    nothing about which offset to use, and none of the recorded answers names
    a vertical side. The frame also covers only about 0.083 m either side of the
    seam, inside the first offset either way.

    So the rule is symmetric. A frame where the inspector named corrosion gets
    the offsets nearest the seam on both sides of it, which is plate just
    beyond the edge of the picture; wall loss is the only thing the probe can
    measure. A weld fault, a sound frame or a missing verdict falls back to on
    the seam.
    """
    if "corrosion" not in named(row):
        return on_the_seam(offsets, station, memory, budget)
    beside = sorted(offsets, key=lambda o: (abs(o) < 1e-6, abs(o), o))
    return beside[:budget]


def sweep_ordered(offsets, station, memory, budget, order):
    """sweep_rule with its tie-break written out as a priority order.

    sweep_rule breaks ties between equally stale offsets by distance from the
    seam, then by list position, which is one order out of 5040. Taking the
    order as an argument is what lets --tiebreaks ask how much of the sweep's
    score is the rule and how much is that choice.
    """
    rank = {o: i for i, o in enumerate(order)}
    picked: list[float] = []
    for offset in memory.thin_offsets(within=2, station=station):
        if offset in offsets and offset not in picked and len(picked) < budget:
            picked.append(offset)

    def staleness(offset: float) -> tuple[int, int]:
        since = memory.stations_since(offset, station)
        return (-(10 ** 6) if since is None else -since, rank[offset])

    for offset in sorted(offsets, key=staleness):
        if len(picked) >= budget:
            break
        if offset not in picked:
            picked.append(offset)
    return picked[:budget]


def shipped_order(offsets) -> list[float]:
    """The tie-break sweep_rule actually uses: nearest the seam, below first."""
    return sorted(offsets, key=abs)


def sweep_models_view(offsets, station, memory, budget, order=None):
    """The sweep, given only what plan_probes puts in front of the model.

    The prompt carries the last six readings, each with its station, offset
    and share of nominal, and a list of offsets never measured on the run. It
    carries no staleness for anything older than those six readings. So this
    arm chases a thin reading inside the window, then takes an offset never
    measured, then one measured only before the window (which it cannot put in
    order, so the tie-break decides), then the stalest inside the window. The
    watch level and the two station window are the rule's own settings, as in
    the sweep; what is held to the model's view is the state.
    """
    order = order or shipped_order(offsets)
    rank = {o: i for i, o in enumerate(order)}
    window = Memory(readings=memory.readings[-SHOWN_READINGS:])
    never = {o for o in offsets if memory.last_seen(o) is None}
    picked: list[float] = []
    for offset in window.thin_offsets(within=2, station=station):
        if offset in offsets and offset not in picked and len(picked) < budget:
            picked.append(offset)

    def priority(offset: float) -> tuple[int, int, int]:
        if offset in never:
            return (0, 0, rank[offset])
        since = window.stations_since(offset, station)
        if since is None:
            return (1, 0, rank[offset])
        return (2, -since, rank[offset])

    for offset in sorted(offsets, key=priority):
        if len(picked) >= budget:
            break
        if offset not in picked:
            picked.append(offset)
    return picked[:budget]


def measure(candidates: list[dict], offset: float) -> dict:
    return next(c for c in candidates if abs(c["offset_m"] - offset) < 1e-6)


def run(frames, answers, chooser, budget, label):
    memory, spent, found, thin_found, rows = Memory(), 0, [], 0, []
    patches: set[str] = set()
    for frame in frames:
        station = frame["station"]
        candidates = frame["probe_candidates"]
        offsets = [c["offset_m"] for c in candidates]
        picked = chooser(offsets, station, memory, budget, frame, answers.get(station))
        for offset in picked:
            c = measure(candidates, offset)
            spent += 1
            memory.readings.append(Reading(station, offset, c["true_thickness_mm"],
                                           c["nominal_mm"]))
            if c["rejectable"]:
                found.append((station, offset, c["true_thickness_mm"]))
                patches.update(c["thinned_by"])
            if c["thinned_by"]:
                thin_found += 1
        rows.append((station, picked))
    return {"label": label, "spent": spent, "rejectable_found": found,
            "thinned_found": thin_found, "patches_found": sorted(patches),
            "rows": rows}


def rejectable_by_station(frames) -> list[tuple[int, int]]:
    """(rejectable points, places) at every station."""
    return [(sum(1 for c in f["probe_candidates"] if c["rejectable"]),
             len(f["probe_candidates"])) for f in frames]


def chance_distribution(frames, budget: int) -> dict:
    """What picking `budget` offsets uniformly at random at every station finds.

    Exact, not simulated. At one station with k rejectable points among n
    places, the number found is hypergeometric; the run is the sum over
    stations, so its distribution is the convolution of theirs. The band is the
    central 95 percent: from the lowest count with more than 2.5 percent at or
    below it to the lowest count with at least 97.5 percent at or below it.
    """
    pmf = [Fraction(1)]
    for k, n in rejectable_by_station(frames):
        take = min(budget, n)
        here = [Fraction(math.comb(k, x) * math.comb(n - k, take - x), math.comb(n, take))
                for x in range(take + 1)]
        out = [Fraction(0)] * (len(pmf) + take)
        for i, p in enumerate(pmf):
            if p:
                for x, q in enumerate(here):
                    out[i + x] += p * q
        pmf = out
    cdf, total = [], Fraction(0)
    for p in pmf:
        total += p
        cdf.append(total)
    lo = next(x for x, c in enumerate(cdf) if c > Fraction(1, 40))
    hi = next(x for x, c in enumerate(cdf) if c >= Fraction(39, 40))
    mean = sum(x * p for x, p in enumerate(pmf))
    return {"what": f"picking {budget} of the places at random at every station, "
                    "exact hypergeometric per station, convolved",
            "mean": float(mean),
            "band_95": [lo, hi],
            "band_mass": float(sum(pmf[lo:hi + 1])),
            "pmf": [float(p) for p in pmf]}


def p_at_least(chance: dict, found: int) -> float:
    """Chance of a random planner finding at least this many."""
    return sum(chance["pmf"][found:])


def oracle_bound(frames, budget: int, answers: dict) -> dict:
    """The most any planner could find, and the most under flagged frames.

    The first is min(budget, rejectable) at every station, summed: a planner
    that knew the answers. The second is the same sum over the frames the
    inspector flagged, which caps anything the camera could steer to even if
    it reported the right side every time.
    """
    flagged = {s for s, row in answers.items() if named(row)}
    best = under = cap = 0
    for frame, (k, _) in zip(frames, rejectable_by_station(frames), strict=True):
        best += min(budget, k)
        if frame["station"] in flagged:
            under += k
            cap += min(budget, k)
    return {"what": f"best possible with {budget} probes a station, knowing the answers",
            "found": best,
            "under_flagged_frames": under,
            "under_flagged_frames_reachable": cap}


def tiebreak_study(frames, answers, budget, rejectable) -> dict:
    """The sweep under every order of its tie-break."""
    offsets = [c["offset_m"] for c in frames[0]["probe_candidates"]]
    shipped = shipped_order(offsets)
    reference = run(frames, answers, lambda o, s, m, b, f, r: sweep_rule(o, s, m, b),
                    budget, "sweep (no model)")
    mine = run(frames, answers,
               lambda o, s, m, b, f, r: sweep_ordered(o, s, m, b, shipped),
               budget, "sweep, shipped order")
    if mine["rows"] != reference["rows"]:
        raise SystemExit("sweep_ordered does not reproduce sweep_rule with the "
                         "shipped tie-break; the study would be about another rule")
    found = []
    for order in itertools.permutations(offsets):
        got = run(frames, answers,
                  lambda o, s, m, b, f, r, order=order: sweep_ordered(o, s, m, b, order),
                  budget, "")
        found.append(len(got["rejectable_found"]))
    as_shipped = len(reference["rejectable_found"])
    rotations = []
    for k in range(len(shipped)):
        order = shipped[k:] + shipped[:k]
        got = run(frames, answers,
                  lambda o, s, m, b, f, r, order=order: sweep_ordered(o, s, m, b, order),
                  budget, "")
        rotations.append(len(got["rejectable_found"]))
    mirrored = [-o if o else 0.0 for o in shipped]
    got = run(frames, answers,
              lambda o, s, m, b, f, r: sweep_ordered(o, s, m, b, mirrored), budget, "")
    counts = {v: found.count(v) for v in sorted(set(found))}
    # The model's view breaks more ties than the sweep does, because it cannot
    # order offsets last measured before its window, so it gets the same study.
    viewed = []
    for order in itertools.permutations(offsets):
        seen = run(frames, answers,
                   lambda o, s, m, b, f, r, order=order:
                   sweep_models_view(o, s, m, b, list(order)),
                   budget, "")
        viewed.append(len(seen["rejectable_found"]))
    return {"orders": len(found), "rejectable": rejectable,
            "shipped_order": shipped, "shipped_found": as_shipped,
            "counts": counts,
            "mean": statistics.fmean(found), "median": statistics.median(found),
            "min": min(found), "max": max(found),
            "share_below_shipped": sum(1 for v in found if v < as_shipped) / len(found),
            "share_at_or_below_shipped": sum(1 for v in found if v <= as_shipped) / len(found),
            "rotations_of_shipped_order": rotations,
            "mirrored_order": mirrored,
            "mirrored_found": len(got["rejectable_found"]),
            "models_view": {"counts": {v: viewed.count(v) for v in sorted(set(viewed))},
                            "mean": statistics.fmean(viewed),
                            "median": statistics.median(viewed),
                            "min": min(viewed), "max": max(viewed)}}


def is_protected(path: Path) -> bool:
    """Whether writing to `path` would replace something already recorded.

    Every existing file under tank/data counts: the recorded probe results, but
    also the answer key, the recorded verdicts, the scene and the frame digests,
    none of which a scoring run has any business replacing, and an earlier
    paid run in runs/. A live run's default name carries the time, so it never
    collides with one.
    """
    resolved = path.resolve()
    return resolved.exists() and resolved.is_relative_to(DATA.resolve())


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=1, default=str))
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(SURVEY / "manifest.json"))
    parser.add_argument("--inspection", default=str(SURVEY / "inspection.jsonl"))
    parser.add_argument("--inspection-model", default=None,
                        help="whose verdicts to read from --inspection, when it "
                             "holds more than one model's")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--models", nargs="*", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true",
                      help="run the model planner. Paid: one Token Factory request "
                           "a station, every repeat. Without this flag nothing here "
                           "calls a model, whatever keys are set")
    mode.add_argument("--baselines", action="store_true",
                      help="the arithmetic planners only, no model and no key. "
                           "The same as leaving out --live, said out loud")
    mode.add_argument("--tiebreaks", action="store_true",
                      help="score the sweep under all orders of its tie-break and "
                           "say where the shipped order falls. No model")
    parser.add_argument("--floor", action="store_true",
                        help="with --live: reserve one probe for the arithmetic "
                             "rule and let the model spend the rest")
    parser.add_argument("--blind", action="store_true",
                        help="with --live: hide the camera from the planner. If the "
                             "score does not move, the vision half contributes "
                             "nothing to the probing on this run")
    parser.add_argument("--repeat", type=int, default=1,
                        help="with --live: run each model this many times and report "
                             "the spread, because at temperature zero it moves")
    parser.add_argument("--out", default=None,
                        help="where to write the result. Without it an arithmetic run "
                             "writes nothing and a live run goes to "
                             "tank/data/survey/runs/probing_<time>.json")
    parser.add_argument("--force", action="store_true",
                        help="allow --out to replace a file already under tank/data")
    args = parser.parse_args()

    # Decided before anything runs, so a refusal costs nothing. A live run that
    # wrote over probing.json would replace the recorded comparison with a new
    # draw from the same model, and the reader would have no way to tell.
    out_path = Path(args.out) if args.out else None
    if out_path is None and args.live:
        out_path = RUNS / f"probing_{time.strftime('%Y%m%d_%H%M%S')}.json"
    if out_path is not None and is_protected(out_path) and not args.force:
        print(f"refusing to write over {out_path}: it is already recorded under "
              "tank/data. Pass --force if replacing it is the point.")
        return 2
    if args.live and not nebius.have_key():
        print("--live needs a Token Factory key and none was found. Set "
              "NEBIUS_API_KEY, or leave out --live for the arithmetic planners.")
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = manifest["frames"]
    path = Path(args.inspection)
    answers = {}
    if path.exists():
        # One inspector's verdicts. Two models in the file would otherwise be
        # merged a station at a time, and every count under "passed frames"
        # and the probe-beside-rust arm would describe an inspector that never
        # ran.
        models_in_file = {json.loads(line).get("model")
                          for line in path.open(encoding="utf-8") if line.strip()}
        if args.inspection_model is None and len(models_in_file) > 1:
            print(f"{path} holds verdicts from {len(models_in_file)} models: "
                  f"{sorted(map(str, models_in_file))}. Name one with "
                  "--inspection-model.")
            return 2
        answers = load_answers(path, args.inspection_model)
        if args.inspection_model is not None and not answers:
            print(f"no recorded verdicts for {args.inspection_model} in {path}")
            return 2

    limit = manifest["acceptance"]["thinning"]["limit_fraction"]
    total = sum(len(f["probe_candidates"]) for f in frames)
    rejectable = sum(1 for f in frames for c in f["probe_candidates"] if c["rejectable"])
    thinned = sum(1 for f in frames for c in f["probe_candidates"] if c["thinned_by"])
    print(f"{len(frames)} stations, {total} places the arm could measure, "
          f"time for {len(frames) * args.budget}.")
    print(f"{thinned} of them are over thinned wall and {rejectable} fail the "
          f"{limit * 100:.0f}% limit.")

    # Counted, not asserted. Where the camera can see wall loss is the
    # scorer's own rule (truth_for, which needs 8 percent of the patch or the
    # frame); "touches" is any visible patch in the frame at all.
    visible_patches = {d["id"] for d in manifest.get("defects", [])
                       if d["kind"] == "thinning" and d["visible"]}
    hidden = sorted({i for f in frames for c in f["probe_candidates"] if c["rejectable"]
                     for i in c["thinned_by"]})
    counts = {"frames_showing_wall_loss": 0, "rejectable_under_them": 0,
              "frames_touching_a_visible_patch": 0, "rejectable_under_those": 0,
              "rejectable_on_a_visible_patch": 0,
              "rejectable_under_passed_frames": 0,
              "rejectable_under_frames_with_nothing_in_view": 0}
    for frame in frames:
        k = sum(1 for c in frame["probe_candidates"] if c["rejectable"])
        worst, seen, _ = truth_for(frame)
        if "corrosion" in seen:
            counts["frames_showing_wall_loss"] += 1
            counts["rejectable_under_them"] += k
        if any(d["kind"] == "thinning" and d["visible_to_a_camera"]
               for d in frame["faults_in_view"]):
            counts["frames_touching_a_visible_patch"] += 1
            counts["rejectable_under_those"] += k
        counts["rejectable_on_a_visible_patch"] += sum(
            1 for c in frame["probe_candidates"]
            if c["rejectable"] and visible_patches & set(c["thinned_by"]))
        row = answers.get(frame["station"])
        if row is not None and not named(row):
            counts["rejectable_under_passed_frames"] += k
        if worst == "none":
            counts["rejectable_under_frames_with_nothing_in_view"] += k
    print(f"The camera can see wall loss in {counts['frames_showing_wall_loss']} frames, "
          f"over {counts['rejectable_under_them']} of the {rejectable} rejectable points. "
          f"A visible patch touches {counts['frames_touching_a_visible_patch']} frames, "
          f"over {counts['rejectable_under_those']} of them.")
    print(f"{counts['rejectable_on_a_visible_patch']} are on a patch a camera can see. "
          f"{counts['rejectable_under_passed_frames']} are under frames the inspector "
          f"passed, {counts['rejectable_under_frames_with_nothing_in_view']} under frames "
          f"with no visible fault in view.")
    print(f"They sit on {len(hidden)} hidden patches: {', '.join(hidden)}.\n")

    chance = chance_distribution(frames, args.budget)
    oracle = oracle_bound(frames, args.budget, answers)

    if args.tiebreaks:
        study = tiebreak_study(frames, answers, args.budget, rejectable)
        print(f"sweep under all {study['orders']} orders of its tie-break, "
              f"{rejectable} rejectable:")
        for value, n in study["counts"].items():
            print(f"  {value:>3} found  {n:>5} orders")
        print(f"  mean {study['mean']:.2f}, median {study['median']:g}, "
              f"range {study['min']} to {study['max']}")
        print(f"  shipped order {study['shipped_order']} finds {study['shipped_found']}: "
              f"{study['share_below_shipped'] * 100:.1f}% of orders find fewer, "
              f"{study['share_at_or_below_shipped'] * 100:.1f}% find as many or fewer")
        print(f"  rotations of the shipped order: {study['rotations_of_shipped_order']}")
        print(f"  mirrored order {study['mirrored_order']} finds {study['mirrored_found']}")
        view = study["models_view"]
        print()
        print(f"sweep, model's view, under the same {study['orders']} orders:")
        for value, n in view["counts"].items():
            print(f"  {value:>3} found  {n:>5} orders")
        print(f"  mean {view['mean']:.2f}, median {view['median']:g}, "
              f"range {view['min']} to {view['max']}")
        if out_path is not None:
            write(out_path, {"budget": args.budget, "places": total,
                             "rejectable": rejectable, "tiebreaks": study})
            print(f"\nwrote {out_path}")
        return 0

    results = []
    for label, chooser in (
        ("on the seam, then -0.15", lambda o, s, m, b, f, r: on_the_seam(o, s, m, b)),
        ("probe beside reported rust",
         lambda o, s, m, b, f, r: where_it_saw(o, s, m, b, r)),
        ("sweep (no model)", lambda o, s, m, b, f, r: sweep_rule(o, s, m, b)),
        ("sweep, model's view", lambda o, s, m, b, f, r: sweep_models_view(o, s, m, b)),
    ):
        results.append(run(frames, answers, chooser, args.budget, label))

    latencies, fallbacks, spreads = [], 0, []
    for model in ((args.models or [planner_model()]) if args.live else []):
        plans: dict[int, object] = {}

        def choose(offsets, station, memory, budget, frame, row, model=model,
                   plans=plans):
            nonlocal fallbacks
            # One nominal for the station, taken from the first candidate. The
            # band straddles a course boundary, so for some offsets this is the
            # wrong plate; the recorded runs were made this way and it is one
            # of the ways the model is shown less than the rule.
            course = frame["probe_candidates"][0]["nominal_mm"]
            planner = plan_with_floor if args.floor else plan_probes
            # The ablation. Same prompt, same memory, same budget, with the
            # one sentence about the picture replaced by a statement that
            # there is no picture. Everything else is held still.
            seen = ("no camera on this run" if args.blind else said(row))
            plan = planner(station, offsets, course, limit, seen,
                           memory, budget=budget, model=model)
            latencies.append(plan.latency_ms)
            fallbacks += int(plan.fell_back)
            plans[station] = plan
            return plan.offsets

        # Repeated, because the same prompt at temperature zero does not
        # produce the same plan twice. On the 28 point run Nano has found 9,
        # 12 and 10 over three recorded runs (probing.json, probing_power.json),
        # and on the retired four point scene it found 4, 3 and 4 while Super
        # found 0, 3 and 0 (archive_four_point). One run of either would
        # support almost any claim, so the spread gets reported and the worst
        # run counts.
        attempts = []
        for index in range(max(1, args.repeat)):
            plans.clear()
            started = time.perf_counter()
            got = run(frames, answers, choose, args.budget,
                      registry.display(model)
                      + (" + rule floor" if args.floor else "")
                      + (" (blind)" if args.blind else ""))
            got["seconds"] = time.perf_counter() - started
            got["why"] = {k: v.why for k, v in plans.items()}
            got["attempt"] = index + 1
            attempts.append(got)
            results.append(got)
        if len(attempts) > 1:
            spreads.append((registry.display(model),
                            sorted(len(a["rejectable_found"]) for a in attempts)))

    print(f"{'planner':>28} {'probes':>7} {'thinned':>9} {'rejectable':>11} "
          f"{'patches':>8} {'chance >=':>10}")
    print("-" * 80)
    for got in results:
        n = len(got["rejectable_found"])
        print(f"{got['label']:>28} {got['spent']:>7} "
              f"{got['thinned_found']:>4}/{thinned:<4} "
              f"{n:>6}/{rejectable:<4} "
              f"{len(got['patches_found']):>4}/{len(hidden):<3} "
              f"{p_at_least(chance, n):>10.4f}"
              f"   {[f'{s}@{o:+.2f}' for s, o, _ in got['rejectable_found']]}")
    print("-" * 80)
    print(f"{'chance, 2 of 7 at random':>28} {len(frames) * args.budget:>7} "
          f"{'':>9} {chance['mean']:>6.1f} expected, central 95% "
          f"{chance['band_95'][0]} to {chance['band_95'][1]} "
          f"(holds {chance['band_mass'] * 100:.1f}%)")
    print(f"{'oracle, knows the answers':>28} {len(frames) * args.budget:>7} "
          f"{'':>9} {oracle['found']:>6}/{rejectable:<4}")
    print(f"{'':>28} under frames the inspector flagged: "
          f"{oracle['under_flagged_frames']} rejectable, "
          f"{oracle['under_flagged_frames_reachable']} reachable with "
          f"{args.budget} a station")
    if spreads:
        print()
        for label, scores in spreads:
            print(f"  {label:>22} over {len(scores)} runs: "
                  f"{', '.join(str(v) for v in scores)} of {rejectable}, "
                  f"worst {scores[0]}, median {statistics.median(scores):g}")
    rule = next((r for r in results if r["label"].startswith("sweep")), None)
    if rule is not None:
        print(f"\n  The sweep is deterministic at {len(rule['rejectable_found'])} of "
              f"{rejectable} for its shipped tie-break order. --tiebreaks scores "
              f"every other order.")
    if latencies:
        print(f"\n  {round(statistics.median(latencies))} ms a station, "
              f"{fallbacks} fell back to the rule")

    if not args.live:
        print()
        print("  Arithmetic planners only. The model arm runs only with --live,")
        print("  which spends a Token Factory request a station.")

    if out_path is None:
        print("\n  Nothing written. Pass --out to keep this result.")
        return 0
    write(out_path,
          {"budget": args.budget, "places": total, "rejectable": rejectable,
           "thinned": thinned, "counts": counts,
           "hidden_rejectable_patches": hidden,
           "planners": [{k: v for k, v in r.items() if k != "rows"} for r in results],
           "chance": chance, "oracle": oracle})
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
