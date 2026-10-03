"""Score a model on the survey run, against the answers written before it looked.

    python tank/score_survey.py
    python tank/score_survey.py --models deepseek-ai/DeepSeek-V4.1-Flash MiniMaxAI/MiniMax-M3

Every frame carries a list of the faults that were actually in front of the
lens, projected through the camera rather than assumed from being somewhere in
the tank. So a verdict is marked three ways and they are reported separately,
because they mean different things on a tank floor.

  pass or defect   the call that decides whether anybody has to come and look.
                   A false alarm sends a person to a sound weld; a miss leaves
                   a fault in a shell that holds product.
  faults found     of the things actually in the frame, how many were named.
                   This is the one that matters after the gate, because a frame
                   can hold a groove in the weld and a corroded plate at once
                   and they are two different jobs for two different people.
  invented         named something that was not in the frame at all.
  headline right   the one fault it put at the top. Kept for continuity with
                   the earlier runs, and read with care: it was ranked by an
                   order borrowed from a production welding cell, where new
                   fabrication makes corrosion on the parent plate the least of
                   anybody's problems. On a shell already in service wall loss
                   is the thing that opens tanks, so a model that headlines a
                   fully corroded plate over a weld-toe groove is not obviously
                   wrong, and for four frames it was marked wrong for exactly
                   that. That is why the column above exists.
  declined         the model would not judge the frame. Neither right nor
                   wrong: a frame somebody has to open themselves.

One model call a frame. Nothing runs until this is invoked.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain import nebius, registry  # noqa: E402
from tank.inspect_seam import inspect_seam, worst  # noqa: E402

SURVEY = ROOT / "tank" / "data" / "survey"

# The scene names a fault for what it is; the inspector names it for what it
# looks like. Wall loss is planted as thinning because that is the measurement
# the probe has to make, and what shows on the surface of it is rust, which the
# inspector is asked to call corrosion. Scored word against word rather than
# thing against thing, every correct call of rust read as an invention.
CAMERA_WORD = {"thinning": "corrosion"}

# How much of a fault has to be in the frame before the frame is about it. A
# patch three quarters of a metre across can show as a sliver in one corner,
# and a model that calls that length of seam sound has not missed anything
# worth walking to. Slivers are neither a defect to find nor an invention to
# punish, so they are held out of both columns and counted on their own.
TRACE = 0.08


def truth_for(frame: dict) -> tuple[str, list[str], list[str]]:
    """What was in front of the lens, the worst of it, and what merely grazed it.

    Only faults a camera could see count against the picture. A patch of wall
    loss with nothing on the surface is in the frame in the sense that the
    crawler was pointing at it, and no camera on earth would find it, so
    marking the model down for missing it would be marking it down for physics.
    """
    visible = [f for f in frame["faults_in_view"] if f["visible_to_a_camera"]]
    seen = [CAMERA_WORD.get(f["kind"], f["kind"]) for f in visible
            if f.get("coverage", 1.0) >= TRACE]
    grazed = [CAMERA_WORD.get(f["kind"], f["kind"]) for f in visible
              if f.get("coverage", 1.0) < TRACE]
    return worst(seen), seen, grazed


def score(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["verdict"] != "uncertain"]
    declined = len(rows) - len(judged)
    clean = [r for r in judged if r["truth"] == "none"]
    faulty = [r for r in judged if r["truth"] != "none"]
    return {
        "frames": len(rows),
        "judged": len(judged),
        "declined": declined,
        "clean_frames": len(clean),
        "faulty_frames": len(faulty),
        "pass_or_defect_right": sum(1 for r in judged
                                    if (r["truth"] == "none") == (r["called"] == "none")),
        "named_right": sum(1 for r in judged if r["called"] == r["truth"]),
        # Every fault that was really in the frame, and whether it was named.
        "faults_present": sum(len(r["in_view"]) for r in judged),
        "faults_found": sum(len(set(r["faults"]) & set(r["in_view"]))
                            for r in judged),
        "invented": sum(len(set(r["faults"]) - set(r["in_view"]) - set(r["grazed"]))
                        for r in judged),
        "frames_all_found": sum(1 for r in judged
                                if set(r["in_view"]) <= set(r["faults"])),
        # Naming a fault that is genuinely in the frame but ranking it below
        # the worst one there is a different mistake from naming a fault that
        # is not there at all, and on a work order it costs a different amount.
        "named_something_present": sum(1 for r in judged if r["called"] != "none"
                                       and r["called"] in r["in_view"]),
        # Naming something the frame really does hold a sliver of is not an
        # invention, so it is not charged as one.
        "false_alarms": sum(1 for r in clean if r["called"] != "none"
                            and r["called"] not in r["grazed"]),
        "called_a_sliver": sum(1 for r in clean if r["called"] != "none"
                               and r["called"] in r["grazed"]),
        "missed": sum(1 for r in faulty if r["called"] == "none"),
        "misnamed": sum(1 for r in faulty
                        if r["called"] != "none" and r["called"] != r["truth"]),
        "named_a_lesser_fault_that_was_there": sum(
            1 for r in faulty if r["called"] != "none" and r["called"] != r["truth"]
            and r["called"] in r["in_view"]),
        "kinds_named": len({r["called"] for r in judged} - {"none"}),
        "median_ms": round(statistics.median([r["latency_ms"] for r in rows] or [0])),
        "retried": sum(1 for r in rows if r["retried_with_room"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(SURVEY / "manifest.json"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(SURVEY / "inspection.jsonl"))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = manifest["frames"][: args.limit] if args.limit else manifest["frames"]
    models = args.models or [nebius.vision_model()]
    root = Path(args.manifest).parent

    planted = Counter()
    for frame in frames:
        planted[truth_for(frame)[0]] += 1
    print(f"{len(frames)} frames from the survey run. What is actually in them: "
          f"{dict(planted)}\n")

    records: list[dict] = []
    for model in models:
        rows: list[dict] = []
        started = time.perf_counter()
        print(f"{registry.display(model)}")
        print(f"{'frame':>6} {'in front of the lens':>22} {'headline':>16} "
              f"{'ok':>4} {'ms':>6}   everything it listed")
        print("-" * 86)
        for frame in frames:
            truth, seen, grazed = truth_for(frame)
            verdict = inspect_seam(root / frame["rgb"], model=model)
            row = {"station": frame["station"], "truth": truth, "in_view": seen,
                   "grazed": grazed, **verdict.as_row()}
            rows.append(row)
            records.append({"model": model, **row})
            mark = ("n/a" if row["verdict"] == "uncertain"
                    else "yes" if row["called"] == truth else "NO")
            print(f"{frame['station']:>6} {truth:>22} {row['called']:>16} "
                  f"{mark:>4} {row['latency_ms']:>6}   "
                  f"{','.join(row['faults']) or '-'}", flush=True)
        got = score(rows)
        seconds = time.perf_counter() - started
        print(f"\n  pass or defect {got['pass_or_defect_right']}/{got['judged']}"
              f"   faults found {got['faults_found']}/{got['faults_present']}"
              f"   invented {got['invented']}"
              f"   every fault found in {got['frames_all_found']}/{got['judged']}"
              f" frames")
        print(f"  headline right {got['named_right']}/{got['judged']}"
              f"   false alarms {got['false_alarms']}"
              f"   missed {got['missed']}"
              f"   misnamed {got['misnamed']}"
              f" (of which {got['named_a_lesser_fault_that_was_there']} named"
              f" something that really was in the frame)"
              f"   declined {got['declined']}")
        if got["called_a_sliver"]:
            print(f"  {got['called_a_sliver']} more named a fault the frame held"
                  f" only a sliver of, charged to neither column")
        print(f"  {got['median_ms']} ms a frame, {seconds:.0f} s for the run\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
