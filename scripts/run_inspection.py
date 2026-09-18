"""Inspect the welded beads and score the verdicts against what was injected.

The cell measures before welding and looks again after it. This is the second
half: each part's bead is photographed and a vision model returns pass or a named
defect with its reasoning.

What makes the number mean anything is that the defects were put there on purpose.
Every verdict is scored against the injector's own record, so a false pass and a
false alarm are counted separately, which is the distinction a shop cares about:
one ships a bad part, the other scraps a good one.

    python scripts/run_inspection.py --manifest data/cap50/manifest.json
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

from brain.inspector import inspect_weld  # noqa: E402
from cell.client import CellClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/cap50/manifest.json")
    parser.add_argument("--limit", type=int, default=0, help="Inspect only the first N parts")
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--api", default=None)
    parser.add_argument("--out", default="data/inspections.jsonl")
    parser.add_argument("--model", default=None,
                        help="Vision model to judge with. Token Factory serves more than "
                             "one; they do not agree, and which one is used is a result.")
    parser.add_argument("--framing", default="rendering", choices=("as_written", "rendering"),
                        help="Which system prompt: as first written, or one that tells the "
                             "model the picture is a rendering and closes the uncertain exit.")
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                        help="Send the bead window at twice the size instead of the whole part.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    rows = manifest["rows"][: args.limit] if args.limit else manifest["rows"]

    client = None
    if not args.no_post:
        client = CellClient(base_url=args.api)
        if not client.healthy():
            print("[inspect] service not reachable, verdicts will be queued to disk")

    started = time.time()
    records = []
    print(f"{'part':>5} {'injected':>14} {'verdict':>9} {'called':>14} {'ok':>4} {'ms':>6}")
    print("-" * 62)

    for row in rows:
        image = root / row["bead_image"]
        if not image.exists():
            continue
        injected = row.get("injected_defect") or "none"
        result = inspect_weld(image, model=args.model, framing=args.framing, crop=args.crop)
        called = result.defect_type.value
        correct = called == injected

        records.append({
            "trial": row["trial"],
            "injected": injected,
            "verdict": result.verdict.value,
            "called": called,
            "correct": correct,
            "severity": result.severity,
            "reasoning": result.reasoning,
            "model": result.model,
            "latency_ms": result.latency_ms,
            # A reasoning model that spent its first budget thinking was asked
            # again with room. Recorded so the report can say which arm needed it.
            "retried_with_room": bool(result.extra.get("retried_with_room")),
            "framing": args.framing,
            "crop": args.crop,
        })
        room = " *" if result.extra.get("retried_with_room") else ""
        print(f"{row['trial']:>5} {injected:>14} {result.verdict.value:>9} "
              f"{called:>14} {'yes' if correct else 'NO':>4} {result.latency_ms:>6.0f}{room}",
              flush=True)

        if client is not None:
            serial = client.create_part(serial=f"P-{row['trial']:06d}")
            if serial:
                client.post_inspection(
                    serial,
                    image_path=str(image),
                    verdict=result.verdict.value,
                    defect_type=called,
                    severity=result.severity,
                    reasoning=result.reasoning,
                    model=result.model,
                    latency_ms=result.latency_ms,
                    injected_defect=injected,
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    correct = sum(1 for r in records if r["correct"])
    clean = [r for r in records if r["injected"] == "none"]
    faulty = [r for r in records if r["injected"] != "none"]
    false_alarm = sum(1 for r in clean if r["called"] != "none")
    missed = sum(1 for r in faulty if r["called"] == "none")

    print(f"\n{correct}/{len(records)} verdicts match what was injected")
    print(f"  clean parts   {len(clean):>3}, called defective {false_alarm} "
          f"(a false alarm scraps a good part)")
    print(f"  faulty parts  {len(faulty):>3}, called clean {missed} "
          f"(a miss ships a bad one)")
    if faulty:
        named = Counter(r["called"] for r in faulty if r["called"] != "none")
        print(f"  defects named: {dict(named)}")
    if records:
        print(f"  median latency {statistics.median(r['latency_ms'] for r in records):.0f} ms")
    print(f"\nwrote {out} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
