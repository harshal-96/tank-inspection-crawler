"""Score the supervisor agent against the database it is answering from.

The supervisor was the one model call in the cell that had never been scored:
it ran, and nobody checked whether what it said matched what the tools said.
Every question here has an answer the database can state exactly, so a reply
is marked right when the number it gives is the number on record, wrong when
it gives another, and unanswered when it gives none. Whether it logged a
decision when it recommended one is checked from the tool calls it made.

Costs about three model calls a question. Nothing runs until it is invoked.

    python scripts/eval_supervisor.py --models nvidia/Nemotron-3_5-Lightning
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session  # noqa: E402

from brain import nebius, supervisor, tools  # noqa: E402
from service.db import engine  # noqa: E402

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(text.replace(",", ""))]


def has_number(text: str, wanted: float, tol: float) -> bool:
    return any(abs(n - wanted) <= tol for n in numbers_in(text))


def truth(session: Session) -> dict[str, Any]:
    status = tools.get_cell_status(session, 20)
    drift = tools.get_drift(session, 20)
    defects = tools.list_defects(session, 10)
    recent = tools.get_recent_parts(session, 1)
    diagnosis = tools.diagnose_drift(session, 30)
    return {"status": status, "drift": drift, "defects": defects, "recent": recent,
            "diagnosis": diagnosis}


def questions(t: dict[str, Any]) -> list[tuple[str, Callable[[str, list[str]], bool | None]]]:
    st = t["status"]
    err = st["correction_error_mm"]
    insp = st["inspection"]
    drift = t["drift"]
    last = (t["recent"].get("parts") or [{}])[0]
    diag = t["diagnosis"]

    def mentions_trend(reply: str, _calls: list[str]) -> bool:
        low = reply.lower()
        if drift.get("available") is False:
            return "not enough" in low or "need" in low
        if drift["trending"]:
            return "trend" in low and "not a trend" not in low and "no trend" not in low
        return any(w in low for w in ("not a trend", "no trend", "scatter", "not drifting",
                                     "no drift", "nothing is drifting"))

    def names_the_cause(reply: str, calls: list[str]) -> bool:
        if not diag.get("available"):
            return None
        cause = diag["cause"]
        low = reply.lower()
        words = {"none": ("nothing", "no machine", "none", "no fault", "not drifting"),
                 "press": ("press",), "fixture": ("fixture", "locator", "clamp")}
        return "diagnose_drift" in calls and any(w in low for w in words.get(cause, ()))

    def no_decision_unless_recommended(reply: str, calls: list[str]) -> bool:
        low = reply.lower()
        recommended = any(w in low for w in ("recommend", "suggest", "should", "hold the line"))
        return ("record_decision" in calls) == recommended or not recommended

    return [
        ("What is the mean correction error over the last 20 parts, in millimetres?",
         lambda r, c: has_number(r, err["mean"], 0.011)),
        ("How many parts have gone through the cell in total?",
         lambda r, c: has_number(r, st["parts_total"], 0)),
        ("How many defects did the inspector miss?",
         lambda r, c: has_number(r, insp["missed_defects"], 0)),
        ("How many false alarms has the inspector raised?",
         lambda r, c: has_number(r, insp["false_alarms"], 0)),
        ("What is the worst correction error we have seen, in millimetres?",
         lambda r, c: has_number(r, err["max"], 0.011)),
        ("Is the joint position drifting, or is it just scatter?", mentions_trend),
        ("Which machine is going out of adjustment, if any?", names_the_cause),
        (f"What was the correction error on part {last.get('serial', '')}?",
         lambda r, c: has_number(r, last.get("error_mm") or 0.0, 0.011)
         if last.get("error_mm") is not None else None),
        ("Should we hold the line right now?", no_decision_unless_recommended),
        ("What fraction of parts landed under a millimetre?",
         lambda r, c: has_number(r, round(100 * err["sub_mm_rate"]), 1)
         or has_number(r, err["sub_mm_rate"], 0.011)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out", default="data/supervisor_eval.jsonl")
    args = parser.parse_args()
    models = args.models or [nebius.text_model()]

    records = []
    with Session(engine) as session:
        t = truth(session)
        qs = questions(t)
        print(f"{len(qs)} questions with answers on record\n")
        for model in models:
            right = wrong = skipped = 0
            calls_total = 0
            started = time.perf_counter()
            for question, check in qs:
                reply = supervisor.ask(question, session, model=model)
                calls = [c.name for c in reply.tool_calls]
                calls_total += reply.rounds
                verdict = check(reply.reply, calls)
                if verdict is None:
                    skipped += 1
                elif verdict:
                    right += 1
                else:
                    wrong += 1
                records.append({"model": model, "question": question, "reply": reply.reply,
                                "tools": calls, "rounds": reply.rounds,
                                "correct": verdict, "latency_ms": reply.latency_ms})
                mark = "  ok " if verdict else (" n/a " if verdict is None else " NO  ")
                print(f"{mark} {question[:60]:<60} {reply.reply[:70]!r}", flush=True)
            secs = time.perf_counter() - started
            print(f"\n{model}: {right} right, {wrong} wrong, {skipped} not scorable, "
                  f"{calls_total} model rounds, {secs:.0f} s\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
