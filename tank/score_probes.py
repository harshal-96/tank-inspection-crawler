"""Score the probe planner on the survey run, against answers written first.

    python tank/score_probes.py
    python tank/score_probes.py --budget 3 --models nvidia/Nemotron-3_5-Lightning

The run offers 168 places to measure and there is time for 48 of them. Four of
the 168 fail the acceptance limit, and all four are under frames the inspector
reads as sound, because the patch they sit on has nothing on the surface. The
patch that does have something on the surface measures inside limits.

That is the whole argument of the machine in one dataset, so it is worth being
blunt about what this measures: not whether a model can see rust, but whether
it will spend a scarce measurement on plate that looks fine.

Three planners are compared on the same stations.

  on the seam    always measures offset 0.0. What a fixed program does.
  where it saw   measures beside whichever side the inspector reported, and on
                 the seam otherwise. What using the camera alone looks like.
  sweep          covers the band in rotation and goes back to a thin reading.
                 Arithmetic, no model, and the thing to beat.
  Nemotron       the same job given to an NVIDIA model, one call a station.

The verdicts come from tank/data/survey/inspection.jsonl, already paid for, so
running this does not call the vision model again.
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

from brain import nebius, registry  # noqa: E402
from tank.probe_planner import (DEFAULT_BUDGET, Memory, Reading, plan_probes,  # noqa: E402
                                planner_model, sweep_rule)

SURVEY = ROOT / "tank" / "data" / "survey"


def said(row: dict) -> str:
    """The inspector's reading of the frame, in the words it used."""
    if row is None:
        return "no inspection of this frame"
    faults = row.get("faults") or ([] if row["called"] == "none" else [row["called"]])
    if not faults:
        return "sound seam, nothing on the surface"
    where = row.get("where") or "not stated"
    return f"{', '.join(faults)} ({where} of the frame)"


def on_the_seam(offsets, station, memory, budget):
    picked = [o for o in offsets if abs(o) < 1e-6]
    return (picked + [o for o in sorted(offsets, key=abs) if o not in picked])[:budget]


def where_it_saw(offsets, station, memory, budget, row=None):
    """Follow the camera: measure toward the side it reported something on."""
    text = (said(row) if row else "").lower()
    if "sound seam" in text or not row:
        return on_the_seam(offsets, station, memory, budget)
    side = -1.0 if ("left" in text or "below" in text) else 1.0
    ranked = sorted(offsets, key=lambda o: (-(o * side), abs(o)))
    return ranked[:budget]


def measure(candidates: list[dict], offset: float) -> dict:
    return next(c for c in candidates if abs(c["offset_m"] - offset) < 1e-6)


def run(frames, answers, chooser, budget, label):
    memory, spent, found, thin_found, rows = Memory(), 0, [], 0, []
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
            if c["thinned_by"]:
                thin_found += 1
        rows.append((station, picked))
    return {"label": label, "spent": spent, "rejectable_found": found,
            "thinned_found": thin_found, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(SURVEY / "manifest.json"))
    parser.add_argument("--inspection", default=str(SURVEY / "inspection.jsonl"))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default=str(SURVEY / "probing.json"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = manifest["frames"]
    answers = {}
    path = Path(args.inspection)
    if path.exists():
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            answers[row["station"]] = row

    limit = manifest["acceptance"]["thinning"]["limit_fraction"]
    total = sum(len(f["probe_candidates"]) for f in frames)
    rejectable = sum(1 for f in frames for c in f["probe_candidates"] if c["rejectable"])
    thinned = sum(1 for f in frames for c in f["probe_candidates"] if c["thinned_by"])
    print(f"{len(frames)} stations, {total} places the arm could measure, "
          f"time for {len(frames) * args.budget}.")
    print(f"{thinned} of them are over thinned wall and {rejectable} fail the "
          f"{limit * 100:.0f}% limit.")
    seen_by_camera = sum(1 for f in frames if any(
        d["kind"] == "thinning" and d["visible_to_a_camera"]
        for d in f["faults_in_view"]))
    print(f"The camera can see wall loss in {seen_by_camera} frames. None of the "
          f"{rejectable} rejectable points are in them.\n")

    results = []
    for label, chooser in (
        ("on the seam", lambda o, s, m, b, f, r: on_the_seam(o, s, m, b)),
        ("where it saw", lambda o, s, m, b, f, r: where_it_saw(o, s, m, b, r)),
        ("sweep (no model)", lambda o, s, m, b, f, r: sweep_rule(o, s, m, b)),
    ):
        results.append(run(frames, answers, chooser, args.budget, label))

    latencies, fallbacks = [], 0
    for model in (args.models or [planner_model()]):
        plans: dict[int, object] = {}

        def choose(offsets, station, memory, budget, frame, row, model=model):
            nonlocal fallbacks
            course = frame["probe_candidates"][0]["nominal_mm"]
            plan = plan_probes(station, offsets, course, limit, said(row), memory,
                               budget=budget, model=model)
            latencies.append(plan.latency_ms)
            fallbacks += int(plan.fell_back)
            plans[station] = plan
            return plan.offsets

        started = time.perf_counter()
        got = run(frames, answers, choose, args.budget, registry.display(model))
        got["seconds"] = time.perf_counter() - started
        got["why"] = {s: p.why for s, p in plans.items()}
        results.append(got)

    print(f"{'planner':>20} {'probes':>7} {'thinned':>9} {'rejectable found':>18}")
    print("-" * 60)
    for got in results:
        print(f"{got['label']:>20} {got['spent']:>7} "
              f"{got['thinned_found']:>4}/{thinned:<4} "
              f"{len(got['rejectable_found']):>9}/{rejectable}"
              f"   {[f'{s}@{o:+.2f}' for s, o, _ in got['rejectable_found']]}")
    if latencies:
        print(f"\n  {round(statistics.median(latencies))} ms a station, "
              f"{fallbacks} fell back to the rule")

    Path(args.out).write_text(json.dumps(
        {"budget": args.budget, "places": total, "rejectable": rejectable,
         "thinned": thinned,
         "planners": [{k: v for k, v in r.items() if k != "rows"} for r in results]},
        indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
