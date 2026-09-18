"""Score the corrective-action briefs the cell raises, from real faults on record.

The brief writer was never scored either. Each brief here is raised from a
fault the cell actually recorded, and marked on what can be checked without a
person: it came back at all, its sources paragraph cites a source it was given,
every figure in that paragraph appears in the cited source's text, the measured
figure it was handed appears in its own account of the measurement, and it says
what to do. Whether the advice is good welding advice is for a welder to judge;
this catches the brief that invents.

Costs one search and one model call a brief. Nothing runs until it is invoked.

    python scripts/eval_briefs.py --limit 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select  # noqa: E402

from brain import advisor, tools  # noqa: E402
from service.db import engine  # noqa: E402
from service.models import Inspection, SeamTrack, Verdict  # noqa: E402

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def requests_from_record(session: Session, limit: int) -> list[tuple[str, dict]]:
    """Faults on record, newest first, each with the figure the brief must carry."""
    out: list[tuple[str, dict]] = []
    tracks = session.exec(select(SeamTrack).order_by(SeamTrack.ts.desc()).limit(50)).all()
    for t in tracks:
        if t.gap_mm_max and t.gap_mm_max >= 0.5:
            out.append(("gap", {"gap_mm": round(t.gap_mm_max, 2), "thickness_mm": 2.0,
                                "station_count": t.stations_measured or 20,
                                "figure": round(t.gap_mm_max, 2)}))
        if len(out) >= limit:
            return out
    defects = session.exec(
        select(Inspection).where(Inspection.verdict == Verdict.DEFECT)).all()
    by_type: dict[str, int] = {}
    for d in defects:
        by_type[d.defect_type] = by_type.get(d.defect_type, 0) + 1
    total = session.exec(select(Inspection)).all()
    for defect, count in sorted(by_type.items(), key=lambda kv: -kv[1]):
        if defect and defect != "none":
            out.append(("defect", {"defect": defect, "count": count, "of": len(total),
                                   "figure": count}))
        if len(out) >= limit:
            return out
    diag = tools.diagnose_drift(session, 30)
    if diag.get("available") and diag["cause"] in ("press", "fixture"):
        out.append(("drift", {"cause": diag["cause"],
                              "slope_mm_per_part": diag["tilt_slope_mm_per_part"]
                              if diag["cause"] == "press" else diag["shift_slope_mm_per_part"],
                              "parts": diag["parts"],
                              "figure": abs(diag["tilt_slope_mm_per_part"]
                                            if diag["cause"] == "press"
                                            else diag["shift_slope_mm_per_part"])}))
    return out[:limit]


def raise_brief(kind: str, spec: dict) -> advisor.Brief:
    if kind == "gap":
        return advisor.brief_for_gap(spec["gap_mm"], spec["thickness_mm"], spec["station_count"])
    if kind == "defect":
        return advisor.brief_for_defect(spec["defect"], spec["count"], spec["of"])
    return advisor.brief_for_drift(spec["cause"], spec["slope_mm_per_part"], spec["parts"])


def score(brief: advisor.Brief, figure: float) -> dict:
    measured_ok = any(abs(float(n) - figure) <= 0.011
                      for n in _NUMBER.findall(brief.what_the_cell_measured.replace(",", "")))
    return {
        "returned": brief.error is None,
        "citations_ok": brief.error is None and brief.citations_ok,
        "measured_figure_present": measured_ok,
        "has_steps": bool(brief.do_now),
        "confidence": brief.confidence,
        "documents_read": brief.documents_read,
        "total_ms": brief.total_ms,
        "error": brief.error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--out", default="data/briefs_eval.jsonl")
    args = parser.parse_args()

    records = []
    with Session(engine) as session:
        asks = requests_from_record(session, args.limit)
    print(f"{len(asks)} briefs to raise from faults on record\n")
    print(f"{'kind':<7} {'back':>5} {'cited':>6} {'figure':>7} {'steps':>6} {'read':>5} {'s':>6}")
    print("-" * 50)
    for kind, spec in asks:
        brief = raise_brief(kind, spec)
        got = score(brief, spec["figure"])
        records.append({"kind": kind, "spec": spec, "score": got, "brief": brief.as_dict()})
        yn = lambda v: "yes" if v else "NO"  # noqa: E731
        print(f"{kind:<7} {yn(got['returned']):>5} {yn(got['citations_ok']):>6} "
              f"{yn(got['measured_figure_present']):>7} {yn(got['has_steps']):>6} "
              f"{got['documents_read']:>5} {got['total_ms'] / 1000:>6.1f}", flush=True)
        if got["error"]:
            print(f"        {got['error'][:100]}")

    n = len(records)
    if n:
        print(f"\ncame back {sum(r['score']['returned'] for r in records)}/{n}, "
              f"cited {sum(r['score']['citations_ok'] for r in records)}/{n}, "
              f"measured figure carried {sum(r['score']['measured_figure_present'] for r in records)}/{n}, "
              f"steps {sum(r['score']['has_steps'] for r in records)}/{n}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
