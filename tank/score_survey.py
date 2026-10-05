"""Score a model on the survey run, against the answers written before it looked.

    python tank/score_survey.py --replay         # scores the stored answers, no key
    python tank/score_survey.py --live           # asks the model again: paid, needs a key,
                                                 # writes runs/inspection_<time>.jsonl
                                                 # under tank/data/survey
    python tank/score_survey.py --live --models deepseek-ai/DeepSeek-V4.1-Flash MiniMaxAI/MiniMax-M3

With neither flag nothing runs, so leaving one out never costs anything.

Every frame carries a list of the faults that were actually in front of the
lens, projected through the camera rather than assumed from being somewhere in
the tank. So a verdict is marked three ways and they are reported separately,
because they mean different things on a tank floor.

  pass or defect   the call that decides whether anybody has to come and look.
                   A false alarm sends a person to a sound weld; a miss leaves
                   a fault in a shell that holds product.
  fault in view    the same call marked strictly: a defect call counts only if
                   the fault it names is really in the frame. A frame holding
                   an undercut, called defect for a sliver of rust, is right
                   on the gate and wrong here.
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

With --live, one model call a frame. Nothing runs until this is invoked.
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
from tank.truth import CAMERA_WORD, TRACE, load_answers, truth_for  # noqa: E402, F401

SURVEY = ROOT / "tank" / "data" / "survey"
RUNS = SURVEY / "runs"
# The verdicts every published survey number was computed from. No run, with
# or without --force, writes over them.
RECORDED = SURVEY / "inspection.jsonl"

# The answer key, and the words it is written in, live in tank/truth.py so that
# this scorer, the dashboard and the tests cannot drift apart. In brief: the
# scene names wall loss "thinning" and a camera calls it corrosion (CAMERA_WORD);
# a fault has to fill TRACE of itself or of the frame to count, and below that
# it is a sliver, charged neither as a miss nor as an invention.


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
        # The same gate, marked strictly. A call of "defect" counts only if the
        # fault it names is really in the frame, and a pass only if the frame
        # is really clean. Frames 1 and 2 are why this exists: both hold an
        # undercut and were called defect for a sliver of rust below TRACE,
        # which is the right walk-in for the wrong reason.
        "gate_right_for_a_fault_in_view": sum(
            1 for r in judged
            if (r["truth"] == "none" and r["called"] == "none")
            or (r["called"] != "none" and r["called"] in r["in_view"])),
        "named_right": sum(1 for r in judged if r["called"] == r["truth"]),
        # Every fault that was really in the frame, and whether it was named.
        "faults_present": sum(len(set(r["in_view"])) for r in judged),
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


def per_fault(frames: list[dict], rows: list[dict]) -> list[dict]:
    """Each fault on the run, and whether any frame that held it named it.

    A frame-by-frame score charges the same crack twice if two frames hold it,
    and forgives a fault the run never named if every frame holding it was
    marked on something else. What a tank owner needs to know is whether each
    defect made it onto the work order at least once, so that is counted too.
    """
    named = {r["station"]: set(r["faults"]) for r in rows
             if r["verdict"] != "uncertain"}
    faults: dict[str, dict] = {}
    for frame in frames:
        if frame["station"] not in named:
            continue
        for f in frame["faults_in_view"]:
            if not f["visible_to_a_camera"] or f.get("coverage", 1.0) < TRACE:
                continue
            word = CAMERA_WORD.get(f["kind"], f["kind"])
            entry = faults.setdefault(f["id"], {"id": f["id"], "kind": word,
                                                "frames": [], "named_in": []})
            entry["frames"].append(frame["station"])
            if word in named[frame["station"]]:
                entry["named_in"].append(frame["station"])
    return list(faults.values())


def replayed_rows(frames: list[dict], answers: dict[int, dict]) -> list[dict]:
    """The recorded verdicts, marked against the key as it stands now.

    `answers` is one recorded row a station, as truth.load_answers returns
    them. Each row comes back with truth, in_view and grazed recomputed from the
    frame, so every reader of the recorded run (this replay, significance.py,
    the tests) marks it by one rule. Frames with no recorded verdict are left
    out rather than guessed at.
    """
    rows = []
    for frame in frames:
        if frame["station"] not in answers:
            continue
        truth, seen, grazed = truth_for(frame)
        row = dict(answers[frame["station"]])
        row.update(truth=truth, in_view=seen, grazed=grazed)
        row.setdefault("faults", [] if row["called"] == "none" else [row["called"]])
        rows.append(row)
    return rows


def key_changes(frames: list[dict], answers: dict[int, dict]) -> list[tuple]:
    """Stations where the truth stored with a verdict differs from the key now.

    Returned as (station, stored truth, stored in_view, truth now, in_view now).
    """
    changed = []
    for frame in frames:
        row = answers.get(frame["station"])
        if row is None or "truth" not in row:
            continue
        truth, seen, _ = truth_for(frame)
        if (row["truth"], sorted(row.get("in_view", []))) != (truth, sorted(seen)):
            changed.append((frame["station"], row["truth"], row.get("in_view", []),
                            truth, seen))
    return changed


def print_summary(got: dict) -> None:
    """The headline counts from score(), in the order the README quotes them."""
    print(f"\n  pass or defect {got['pass_or_defect_right']}/{got['judged']}"
          f"   faults found {got['faults_found']}/{got['faults_present']}"
          f"   invented {got['invented']}"
          f"   every fault found in {got['frames_all_found']}/{got['judged']} frames")
    print(f"  right call for a fault actually in view "
          f"{got['gate_right_for_a_fault_in_view']}/{got['judged']}"
          f"   (a defect call counts only if what it names is in the frame)")
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


def replay(frames: list[dict], path: Path, model: str | None = None) -> int:
    """Score answers that were already paid for.

    The verdicts in the file came from real calls on these exact frames. This
    recomputes the truth from the manifest and marks them again, so anybody can
    check the arithmetic without an account. It does not re-ask the model, and
    the summary says so, because "the scoring is right" and "the model is right"
    are different claims and running them together would blur the two.

    The file also holds the truth each verdict was scored against when it was
    recorded. If that differs from the key as it stands now, the difference is
    printed rather than quietly overwritten: an answer key that changes after
    the verdicts are in is the first thing a careful reader should be shown.
    """
    if not path.exists():
        print(f"no recorded answers at {path}")
        return 1
    # One model's verdicts at a time. Mixing two in one score would report a
    # run that no single model made.
    models_in_file = {json.loads(line).get("model")
                      for line in path.open(encoding="utf-8") if line.strip()}
    if model is None and len(models_in_file) > 1:
        print(f"{path} holds verdicts from {len(models_in_file)} models: "
              f"{sorted(map(str, models_in_file))}. Name one with --models.")
        return 1
    answers = load_answers(path, model)
    if not answers:
        # A scorecard of 0/0 reads like a result. It is not one.
        print(f"no recorded verdicts for {model or 'any model'} in {path}; "
              f"nothing to score")
        return 1
    rows = replayed_rows(frames, answers)
    disagree = key_changes(frames, answers)

    print(f"{'frame':>6} {'in front of the lens':>22} {'headline':>16} {'ok':>4}"
          f"   everything it listed")
    print("-" * 78)
    for row in rows:
        mark = ("n/a" if row["verdict"] == "uncertain"
                else "yes" if row["called"] == row["truth"] else "NO")
        print(f"{row['station']:>6} {row['truth']:>22} {row['called']:>16} {mark:>4}"
              f"   {','.join(row['faults']) or '-'}")

    print_summary(score(rows))
    faults = per_fault(frames, rows)
    print(f"  faults on the run named in at least one frame that held them: "
          f"{sum(1 for f in faults if f['named_in'])}/{len(faults)}")
    for f in faults:
        if not f["named_in"]:
            print(f"    never named: {f['id']} {f['kind']}, in frames "
                  f"{', '.join(map(str, f['frames']))}")
    if disagree:
        print(f"\n  The key used when these answers were recorded differs from the key"
              f" now, at {len(disagree)} frame(s):")
        for station, was, was_seen, now, now_seen in disagree:
            print(f"    station {station}: recorded against {was} {was_seen},"
                  f" now {now} {now_seen}")
        print("  The marks above use the key as it is now. See tank/truth.py for why"
              " it moved.")
    print("\n  Replayed from answers already recorded against these frames. This "
          "checks the\n  scoring, not the model. --live asks it again, which is "
          "paid and needs a\n  Token Factory key.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(SURVEY / "manifest.json"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--answers", default=str(RECORDED),
                        help="the recorded verdicts that --replay marks")
    # A live run writes beside the recorded verdicts, never over them. Those
    # are the answers every published number was computed from, and re-asking
    # the model is a new measurement, not a correction of the old one. The
    # default is a new timestamped file in runs/, which git ignores, so one
    # paid rerun cannot wipe out another.
    parser.add_argument("--out", default=None,
                        help="where a --live run writes its verdicts. Default "
                             "tank/data/survey/runs/inspection_<time>.jsonl")
    parser.add_argument("--force", action="store_true",
                        help="allow --out to replace a file that already exists. "
                             "Never the recorded inspection.jsonl")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--replay", action="store_true",
                      help="score the answers already in the jsonl instead of "
                           "asking the model again. No API key needed. It "
                           "checks the scoring, not the model")
    mode.add_argument("--live", action="store_true",
                      help="ask the model again. Paid: one Token Factory request a "
                           "frame for every model named")
    args = parser.parse_args()

    # Without a flag nothing is asked. A plain run that quietly started paid
    # calls whenever a key happened to be in .env would be the wrong default.
    if not (args.replay or args.live):
        print("nothing run: --replay scores the recorded answers, --live asks "
              "the model again (paid)")
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frames = manifest["frames"][: args.limit] if args.limit else manifest["frames"]
    models = args.models or [nebius.vision_model()]
    root = Path(args.manifest).parent

    planted = Counter()
    for frame in frames:
        planted[truth_for(frame)[0]] += 1
    print(f"{len(frames)} frames from the survey run. What is actually in them: "
          f"{dict(planted)}\n")

    if args.replay:
        if args.models and len(args.models) > 1:
            print("--replay marks one model's verdicts at a time; name one.")
            return 1
        return replay(frames, Path(args.answers),
                      args.models[0] if args.models else None)

    out = (Path(args.out) if args.out
           else RUNS / f"inspection_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    if out.resolve() in {RECORDED.resolve(), Path(args.answers).resolve()}:
        print(f"refusing to write a new run over the recorded verdicts in {out}."
              f" Pass a different --out.")
        return 1
    if out.exists() and not args.force:
        print(f"refusing to write over {out}: it already exists. Pass --force if"
              f" replacing it is the point.")
        return 1
    if not nebius.have_key():
        print("No Token Factory key found, so nothing can be asked. To score the"
              " answers already recorded, run with --replay.")
        return 1

    # Imported only here, so that --replay and a refused run never load it.
    from tank.inspect_seam import inspect_seam

    # Each verdict goes to disk as soon as it is back. A transport error at
    # frame twenty would otherwise throw away nineteen answers already paid for.
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
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
                handle.write(json.dumps({"model": model, **row}) + "\n")
                handle.flush()
                mark = ("n/a" if row["verdict"] == "uncertain"
                        else "yes" if row["called"] == truth else "NO")
                print(f"{frame['station']:>6} {truth:>22} {row['called']:>16} "
                      f"{mark:>4} {row['latency_ms']:>6}   "
                      f"{','.join(row['faults']) or '-'}", flush=True)
            got = score(rows)
            seconds = time.perf_counter() - started
            print_summary(got)
            print(f"  {got['median_ms']} ms a frame, {seconds:.0f} s for the run\n")

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
