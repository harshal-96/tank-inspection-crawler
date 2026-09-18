"""Turning a fault the cell measured into an instruction somebody can act on.

The cell is good at two things and bad at a third. It can say the joint opened to
4.25 mm at the third station on a 2 mm sheet, and it can say the press brake angle
has been walking for eleven parts. What it cannot say is what to do about either,
because that answer is not in the measurement. It is in a welding code, a machine
handbook or a process guideline, it changes, and a model's recollection of it is
the least reliable thing in the building.

So the brief is built in three moves, and each one is somebody else's job:

  the cell           writes the question, out of numbers it actually measured. No
                     model is asked what is wrong; the measurement already says.
  Tavily             finds the documents and fetches their text, so what gets
                     quoted is the clause rather than a search snippet.
  Nemotron           writes the brief, from that text only, with a citation on
                     every claim and an explicit refusal when the sources do not
                     answer.

The rule the prompt enforces is the one that matters on a shop floor: a number
that came from the cell is labelled as measured, a number that came from a
document is labelled with the document, and anything else is left out. An
operator who cannot tell those apart cannot act on either.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from brain import lookup, nebius

MAX_TOKENS = 900

SYSTEM_PROMPT = """You write short corrective action briefs for a welding shop.

You are given measurements a robotic cell actually took, and the text of documents
a search returned. Write the brief from those two things and nothing else.

Rules, in order of importance.

1. Never invent a number, a clause number, a standard designation or a limit. If
   the sources do not contain it, say the sources do not contain it.
2. Mark where every number came from. A number the cell measured is "measured".
   A number from a document is followed by [1], [2] and so on, matching the
   numbered sources you were given.
3. The measurement is not up for debate. Do not second-guess what the cell found;
   advise on what to do about it.
4. Say what to do first, physically, by somebody standing at a machine. Not
   "review the process". Something a person can go and do in the next ten minutes.
5. Short. An operator reads this between parts.

Answer with JSON only:
{"headline": "one line naming the fault and the machine to touch",
 "what_the_cell_measured": "one or two sentences, numbers included",
 "what_the_sources_say": "one or two sentences, every claim carrying [n]",
 "do_now": ["two to four steps, each an action at a machine"],
 "confidence": "high" | "medium" | "low",
 "caveat": "what the sources did not settle, or empty string if they settled it"}"""


@dataclass
class Brief:
    """A corrective action brief, with everything needed to check it."""

    topic: str
    query: str
    headline: str = ""
    what_the_cell_measured: str = ""
    what_the_sources_say: str = ""
    do_now: list[str] = field(default_factory=list)
    confidence: str = "low"
    caveat: str = ""
    sources: list[dict[str, str]] = field(default_factory=list)
    model: Optional[str] = None
    documents_read: int = 0
    search_ms: float = 0.0
    total_ms: float = 0.0
    error: Optional[str] = None
    # False when the sources paragraph cites nothing, or cites a source that
    # was not given. The prompt asks for a citation on every claim; this is
    # the check that it happened, because a brief nobody can trace is advice.
    citations_ok: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_CITATION = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")


def check_citations(brief: "Brief", texts: Optional[dict[int, str]] = None) -> "Brief":
    """Mark a brief whose sources paragraph cannot be traced to a given source.

    Two checks. Every claim must carry a citation to a source that was given.
    And, when the source texts are in hand, every figure in the paragraph must
    appear in the text of a cited source, because a figure with a bracket after
    it that the source does not contain is an invented number wearing a badge.
    """
    text = brief.what_the_sources_say
    if not text:
        return brief
    cited = {int(n) for n in _CITATION.findall(text)}
    valid = set(range(1, len(brief.sources) + 1))
    problems: list[str] = []
    if not cited or not cited <= valid:
        problems.append("a missing or invalid citation")
    if texts and cited:
        pool = " ".join(texts.get(n, "") for n in cited if n in valid)
        pool = pool.replace(",", "")
        figures = _NUMBER.findall(_CITATION.sub("", text))
        missing = sorted({f for f in figures if f.replace(",", "") not in pool})
        if missing:
            problems.append("figures the cited sources do not contain: " + ", ".join(missing))
    if problems:
        brief.citations_ok = False
        brief.confidence = "low"
        note = ("The sources paragraph carries " + " and ".join(problems)
                + "; check it against the numbered sources before acting on it.")
        brief.caveat = f"{brief.caveat} {note}".strip()
    return brief


# ------------------------------------------------------------------ the questions
#
# Built from measurements rather than from a model's reading of them. The cell
# already knows what is wrong; what it needs is what a published source says to do.


def gap_question(gap_mm: float, thickness_mm: float, station_count: int) -> tuple[str, str]:
    return (
        "joint opened too far to weld",
        f"root opening {gap_mm:.1f} mm on {thickness_mm:.0f} mm sheet steel fillet weld: "
        f"maximum root gap allowed before backing or buildup is required, and how to "
        f"correct sheet metal fit-up that opens along the joint",
    )


def drift_question(cause: str, slope_mm_per_part: float, parts: int) -> tuple[str, str]:
    if cause == "press":
        return (
            "press brake angle drifting",
            f"press brake bend angle drifting {slope_mm_per_part:.3f} mm per part over "
            f"{parts} parts on sheet steel: causes of progressive bend angle change and "
            f"how to check and reset ram depth, crowning and tooling wear",
        )
    if cause == "fixture":
        return (
            "fixture creeping",
            f"welding fixture locator creeping {slope_mm_per_part:.3f} mm per part over "
            f"{parts} parts: causes of progressive locator movement and how to check "
            f"clamp force, locator wear and thermal growth in a weld fixture",
        )
    return (
        "no fault found",
        "distinguishing normal part-to-part variation from process drift in sheet "
        "metal fabrication statistical process control",
    )


def defect_question(defect: str, count: int, of: int) -> tuple[str, str]:
    spelled = defect.replace("_", " ")
    return (
        f"{spelled} in the bead",
        f"{spelled} in GMAW fillet welds on thin sheet steel: causes, and the welding "
        f"parameter changes that correct it",
    )


# ------------------------------------------------------------------------ building


def write_brief(topic: str, query: str, measured: str,
                model: Optional[str] = None,
                max_results: int = 4) -> Brief:
    """Search, read the documents, and write the brief from them.

    Returns a Brief carrying its error rather than raising. A cell that stops
    running because a standards website is slow is worse than one that says it
    could not check and carries on measuring.
    """
    started = time.perf_counter()
    brief = Brief(topic=topic, query=query)

    found, documents = lookup.read(query, max_results=max_results)
    brief.search_ms = round(found.latency_ms, 1)
    if found.error:
        brief.error = found.error
        brief.total_ms = round((time.perf_counter() - started) * 1000, 1)
        return brief
    if not found.sources:
        brief.error = "the search returned nothing to read"
        brief.total_ms = round((time.perf_counter() - started) * 1000, 1)
        return brief

    brief.sources = [{"n": str(i + 1), "title": s.title, "url": s.url}
                     for i, s in enumerate(found.sources)]
    brief.documents_read = len(documents)

    blocks = []
    texts: dict[int, str] = {}
    for i, source in enumerate(found.sources, start=1):
        body = documents.get(source.url) or source.snippet
        texts[i] = body
        blocks.append(f"[{i}] {source.title}\n{source.url}\n{body}")
    corpus = "\n\n".join(blocks)

    user = (
        f"What the cell measured:\n{measured}\n\n"
        f"Documents the search returned, numbered:\n\n{corpus}\n\n"
        f"Write the brief."
    )

    brief.model = model or nebius.text_model()
    try:
        reply = nebius.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user}],
            model=brief.model, max_tokens=MAX_TOKENS, reasoning_effort="none",
            parses=nebius.extract_json)
        raw = reply.raw or ""
    except Exception as exc:  # noqa: BLE001
        brief.error = f"{type(exc).__name__}: {exc}"
        brief.total_ms = round((time.perf_counter() - started) * 1000, 1)
        return brief

    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        brief.error = "the brief did not come back as JSON"
        brief.total_ms = round((time.perf_counter() - started) * 1000, 1)
        return brief
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        brief.error = f"unparsed brief: {exc}"
        brief.total_ms = round((time.perf_counter() - started) * 1000, 1)
        return brief

    brief.headline = str(data.get("headline") or "").strip()
    brief.what_the_cell_measured = str(data.get("what_the_cell_measured") or "").strip()
    brief.what_the_sources_say = str(data.get("what_the_sources_say") or "").strip()
    steps = data.get("do_now") or []
    brief.do_now = [str(s).strip() for s in steps if str(s).strip()][:5]
    brief.confidence = str(data.get("confidence") or "low").strip().lower()
    brief.caveat = str(data.get("caveat") or "").strip()
    check_citations(brief, texts)
    brief.total_ms = round((time.perf_counter() - started) * 1000, 1)
    return brief


def brief_for_gap(gap_mm: float, thickness_mm: float, station_count: int,
                  part_serial: str = "") -> Brief:
    topic, query = gap_question(gap_mm, thickness_mm, station_count)
    measured = (
        f"The cell tracked the seam at twenty stations before welding. "
        f"{station_count} of them opened wider than one pass can bridge; the widest "
        f"measured {gap_mm:.2f} mm on {thickness_mm:.0f} mm sheet. "
        f"The cell refused those stations rather than laying a bead across them"
        + (f" (part {part_serial})." if part_serial else ".")
    )
    return write_brief(topic, query, measured)


def brief_for_drift(cause: str, slope_mm_per_part: float, parts: int,
                    evidence: str = "") -> Brief:
    topic, query = drift_question(cause, slope_mm_per_part, parts)
    measured = (
        f"Across {parts} parts the cell measured the seam at the bottom and the top of "
        f"the joint on every one. The diagnosis is '{cause}', from a trend of "
        f"{slope_mm_per_part:+.4f} mm per part."
        + (f" {evidence}" if evidence else "")
    )
    return write_brief(topic, query, measured)


def brief_for_defect(defect: str, count: int, of: int) -> Brief:
    topic, query = defect_question(defect, count, of)
    measured = (
        f"Post-weld inspection called {count} of {of} beads "
        f"{defect.replace('_', ' ')} in this run."
    )
    return write_brief(topic, query, measured)
