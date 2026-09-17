"""Letting the supervisor look something up before it gives an operator an answer.

The cell knows a great deal about this part and nothing at all about the wider
world. It can say a gap measured 2.4 mm on 2 mm sheet; it cannot say what a code
permits, what a filler is rated for, or whether a defect is a repair or a scrap.
Those answers live in standards and supplier data, they change, and a model's
recollection of them is the least trustworthy thing in the building.

So when the question is about the world rather than about the part, the supervisor
searches instead of remembering, and cites what it found. An operator can then
check the source, which is the entire difference between a tool that helps on the
shop floor and one nobody is allowed to act on.

Deliberately narrow. This is not a research agent: it answers welding and
materials questions with sources attached, and it is never allowed to be the thing
that decides whether a part passes. The measurement decides that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

# Same as the model client: the key lives in .env, which is gitignored.
load_dotenv()

TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_S = 20.0
MAX_RESULTS = 4

# No topic hint is appended to the query. An earlier version bolted
# "welding fabrication sheet metal" onto every search to keep it on subject, and
# it did the opposite: the search engine read it as intent to find a fabrication
# shop and returned a Better Business Bureau listing and a competitor analysis for
# a welding company. The technical question was already specific; padding it made
# it vague. Depth does the steering instead.
SEARCH_DEPTH = "advanced"


class MissingTavilyKey(RuntimeError):
    """Raised when a lookup is attempted without a key configured."""


@dataclass
class Source:
    title: str
    url: str
    snippet: str

    def cite(self) -> str:
        return f"{self.title} <{self.url}>"


@dataclass
class Lookup:
    """What came back, with where it came from."""

    query: str
    answer: Optional[str] = None
    sources: list[Source] = field(default_factory=list)
    error: Optional[str] = None
    latency_ms: float = 0.0

    def as_tool_result(self) -> dict[str, Any]:
        """Shape the supervisor's tool interface expects."""
        if self.error:
            return {"query": self.query, "error": self.error}
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [{"title": s.title, "url": s.url, "snippet": s.snippet}
                        for s in self.sources],
            "note": "Cite these sources to the operator. Do not present this as "
                    "a measurement from the cell.",
        }


def api_key() -> str:
    key = (os.getenv("TAVILY_API_KEY") or "").strip()
    if not key:
        raise MissingTavilyKey(
            "TAVILY_API_KEY is not set. Lookups are optional: the cell measures "
            "and decides without them."
        )
    return key


def search(query: str, max_results: int = MAX_RESULTS,
           timeout_s: float = DEFAULT_TIMEOUT_S) -> Lookup:
    """Search for something the cell cannot measure.

    Returns a Lookup carrying the error rather than raising, because a supervisor
    that falls over when the internet is slow is worse than one that says it could
    not check.
    """
    import time

    started = time.perf_counter()
    try:
        key = api_key()
    except MissingTavilyKey as exc:
        return Lookup(query=query, error=str(exc))

    payload = {
        "api_key": key,
        "query": query,
        "search_depth": SEARCH_DEPTH,
        "include_answer": True,
        "max_results": max_results,
    }
    try:
        response = httpx.post(TAVILY_URL, json=payload, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # network, auth, shape: all the same to the caller
        return Lookup(query=query, error=f"{type(exc).__name__}: {exc}",
                      latency_ms=(time.perf_counter() - started) * 1000)

    sources = [
        Source(
            title=str(item.get("title") or "").strip(),
            url=str(item.get("url") or "").strip(),
            snippet=str(item.get("content") or "").strip()[:400],
        )
        for item in (data.get("results") or [])
    ]
    return Lookup(
        query=query,
        answer=(data.get("answer") or "").strip() or None,
        sources=sources,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def available() -> bool:
    """Whether lookups can be attempted at all."""
    try:
        api_key()
        return True
    except MissingTavilyKey:
        return False
