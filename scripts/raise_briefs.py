"""Turn the faults this run found into cited corrective action briefs.

The cell already knows what is wrong. It refused stations on some parts because
the joint opened past what one pass can bridge; it diagnosed which machine is
drifting; the post-weld eye named defects. None of that tells anybody what to do,
and the answer is not in the measurement: it is in a welding code, a machine
handbook or a process guideline, which change and which a model should not be
recalling from memory.

So this walks the faults, searches for each one, reads the documents that come
back rather than their snippets, and has Nemotron write the instruction with a
citation on every claim. The result goes into the same database as everything
else, because a brief nobody can trace back to a source is a brief nobody on a
shop floor is allowed to act on.

    python scripts/raise_briefs.py
    python scripts/raise_briefs.py --only gap
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import advisor  # noqa: E402
from cell.client import CellClient  # noqa: E402
from cell.geometry import PART_T  # noqa: E402


def post(client: CellClient | None, brief: advisor.Brief, trigger: str,
         serial: str | None = None) -> None:
    if client is None:
        return
    payload = brief.as_dict()
    payload["trigger"] = trigger
    if serial:
        payload["part_serial"] = serial
    try:
        client.post_brief(payload)
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not store brief: {type(exc).__name__}: {exc}")


def show(brief: advisor.Brief) -> None:
    if brief.error:
        print(f"  [failed] {brief.error}")
        return
    print(f"  {brief.headline}")
    print(f"  read {brief.documents_read} documents, "
          f"{len(brief.sources)} sources, {brief.total_ms / 1000:.1f} s, "
          f"confidence {brief.confidence}")
    for step in brief.do_now:
        print(f"    - {step}")


def widest_gap(schedules: list[dict]) -> tuple[float, int, str | None]:
    """The worst fit-up in the run, and how many stations it cost."""
    worst_gap, worst_count, worst_serial = 0.0, 0, None
    for schedule in schedules:
        refused = [s for s in schedule.get("stations", []) if not s.get("weldable")]
        if not refused:
            continue
        gap = max(s["gap_mm"] for s in refused)
        if gap > worst_gap:
            worst_gap = gap
            worst_count = len(refused)
            worst_serial = schedule.get("part_serial")
    return worst_gap, worst_count, worst_serial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default=None)
    parser.add_argument("--only", choices=["gap", "drift", "defect"], default=None)
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--drift-eval", default="data/drift_eval_nemotron-3-super-120b-a12b.jsonl")
    parser.add_argument("--inspections", default="data/inspections.jsonl")
    args = parser.parse_args()

    client = None if args.no_post else CellClient(base_url=args.api)
    if client is not None and not client.healthy():
        print("[briefs] service not reachable; briefs will be printed, not stored")
        client = None

    started = time.time()
    raised = 0

    # ---- fit-up: the joint the arc was not allowed to bridge
    if args.only in (None, "gap"):
        schedules = client.get_weld_schedules() if client is not None else []
        gap, count, _ = widest_gap(schedules)
        if gap > 0:
            print(f"\n[fit-up] widest refused gap {gap:.2f} mm on {PART_T * 1000:.0f} mm sheet")
            brief = advisor.brief_for_gap(gap, PART_T * 1000, count)
            show(brief)
            post(client, brief, "gap")
            raised += 1
        else:
            print("\n[fit-up] no refused stations in this run, nothing to advise on")

    # ---- drift: which machine is going out
    if args.only in (None, "drift"):
        path = Path(args.drift_eval)
        if path.is_file():
            rows = [json.loads(line) for line in
                    path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            found = next((r for r in rows if r.get("correct") and r.get("said") != "none"), None)
            if found:
                cause = found["said"]
                slope = (found.get("tilt_slope_mm_per_part") if cause == "press"
                         else found.get("shift_slope_mm_per_part")) or 0.0
                print(f"\n[drift] diagnosis '{cause}', {slope:+.4f} mm per part")
                brief = advisor.brief_for_drift(cause, slope, 30, found.get("evidence", ""))
                show(brief)
                post(client, brief, "drift")
                raised += 1
        else:
            print(f"\n[drift] {path} not found, skipping")

    # ---- defects: what the post-weld eye kept calling
    if args.only in (None, "defect"):
        path = Path(args.inspections)
        if path.is_file():
            rows = [json.loads(line) for line in
                    path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
            called = Counter(r["called"] for r in rows if r.get("called") not in (None, "none"))
            if called:
                defect, count = called.most_common(1)[0]
                print(f"\n[defect] most called: {defect} on {count} of {len(rows)} beads")
                brief = advisor.brief_for_defect(defect, count, len(rows))
                show(brief)
                post(client, brief, "defect")
                raised += 1
        else:
            print(f"\n[defect] {path} not found, skipping")

    print(f"\nraised {raised} brief(s) in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
