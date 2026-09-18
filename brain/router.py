"""Which model answers the per-part question: is this measurement good enough to weld to.

One model answers it, the smallest. That is not a shortcut, it is the measured
result. Scored against ground truth on decisions where the simulation knows
whether another look was worth taking, the smallest Nemotron gets 88 of 89 right
under the prompt the cell runs, and so do the 3.5 Lightning and the 550B. The
120B asks for a look it does not need on a third of the parts. There is nothing
for a larger model to add, so nothing is sent to one.

An earlier version escalated conflicting evidence to the 120B. It never once
changed an outcome for the better, and the 120B is the size that scores worst
on this question, so escalating to it would have cost robot moves. The route
still records what the evidence looked like, because an operator asking why a
part took four looks deserves to see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from brain.viewplanner import spread_mm

# The four sizes, smallest first. Every one of these is an NVIDIA open model
# served on Token Factory; all four are scored on the same decisions.
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"

SIZES = (NANO, LIGHTNING, SUPER, ULTRA)

DEFAULT_MODEL = NANO


@dataclass
class Route:
    """Which model was chosen for one decision, and what made it that one."""

    model: str
    tier: str
    difficulty: str
    reasons: list[str] = field(default_factory=list)

    def describe(self) -> str:
        why = "; ".join(self.reasons) if self.reasons else "nothing unusual"
        return f"{self.tier} ({self.difficulty}): {why}"


def _signal(attempt: dict[str, Any], name: str) -> Optional[float]:
    value = (attempt.get("signals") or {}).get(name)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def assess(attempts: list[dict[str, Any]]) -> Route:
    """Name the model for this decision, and record what the evidence looked like.

    The answer is always the smallest model, because that is what scoring
    against ground truth supports. The record is for the operator: how many
    looks, how many refused, how far the views agree.
    """
    if not attempts:
        return Route(DEFAULT_MODEL, "nano", "unknown", ["no looks taken yet"])
    failures = [a for a in attempts if a.get("failed")]
    usable = [a for a in attempts if not a.get("failed")]
    detail = f"{len(attempts)} look(s)"
    if failures:
        detail += f", {len(failures)} refused"
    spread = spread_mm(usable)
    if spread is not None:
        detail += f", views agree to {spread:.2f} mm"
    return Route(DEFAULT_MODEL, "nano", "ordinary", [detail])


def tier_of(model: str) -> str:
    return {NANO: "nano", LIGHTNING: "lightning",
            SUPER: "super", ULTRA: "ultra"}.get(model, "unknown")
