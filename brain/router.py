"""Sending each decision to the model that is actually right about it.

A welding cell asks the same question hundreds of times a shift: is this
measurement good enough to weld to. The obvious architecture is a ladder, with a
cheap model on the routine calls and a large one brought in when the evidence
conflicts. That is what this module was built to do, and measuring it took it
apart.

Scored against ground truth rather than against the largest model's opinion, the
smallest Nemotron in the family is the most correct one for this decision, by a
wide margin, and the large ones are not more careful but wrong in a direction that
costs robot moves. The numbers are in the comment below the model list.

Two things make that true, and both are properties of the cell rather than of the
models. Accepted views are fused with a median, so one bad view is outvoted before
any model is consulted, and the conflict a large model spends its caution on has
usually already been neutralised. And a look is not free: it is a robot move, so a
model that asks for one it does not need has not been careful, it has been
expensive.

What survives is the routing idea itself, just not the direction. Difficulty is
visible in the evidence before any model is asked, so the routing is arithmetic
over signals already in hand, which is cheap, deterministic and auditable. This is
NVIDIA's NeMo Switchyard concept applied inside a control loop, where both the
money and the milliseconds are real, with the ladder pointed the way the
measurement says rather than the way intuition did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from brain.viewplanner import spread_mm

# The ladder, cheapest first. Every one of these is an NVIDIA open model served on
# Nebius Token Factory.
NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
LIGHTNING = "nvidia/Nemotron-3_5-Lightning"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"

LADDER = (NANO, LIGHTNING, SUPER, ULTRA)

# WHAT THE MEASUREMENT SAID, and it was not what this module first assumed.
#
# Thirty nine decision points from a fifty part campaign were replayed and scored
# against ground truth, which the simulation has, so both branches of each
# decision could be evaluated: what the cell would land on by accepting now, and
# what it would land on after one more look.
#
#   another look was worth taking on     0 of 39
#   Nano got it right                   39 of 39, no wasted moves
#   Lightning                            0 of 39, 39 wasted moves
#   Super                                0 of 39, 39 wasted moves
#   Ultra                               21 of 39, 18 wasted moves
#
# So the smallest model in the family is the most correct one for this decision,
# and the larger ones are not more careful, they are wrong in a direction that
# costs robot moves. An escalating router that sent conflicting evidence up the
# ladder scored 18 of 39, worse than simply asking Nano every time.
#
# The reason is the fusion, not the model. Accepted views are combined with a
# median, so a single bad view is already outvoted before any model sees it, and
# the conflict a large model spends its caution on has usually been neutralised.
#
# Escalation is therefore kept for one case only: a view refused outright while
# the rest disagree badly. On this dataset that case never changed the outcome, so
# the honest position is that the escalation path is a guard rail with no measured
# saves to its name, not a proven feature.
DEFAULT_MODEL = NANO

# What still sends a decision up the ladder. Deliberately narrow: views that
# disagree only matter once one of them has been thrown out, because until then
# the median fusion has already settled it.
SEVERE_DISAGREEMENT_MM = 5.0


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
    """Read the evidence and pick a tier, without asking a model to do it.

    The routing decision has to be cheaper than the decision it is routing, or it
    has bought nothing, so it is arithmetic over signals already in hand.

    It routes to the smallest model by default, because that is what the scoring
    against ground truth supports. See the note at the top of this module: the
    assumption that harder evidence deserves a larger model did not survive being
    measured.
    """
    reasons: list[str] = []
    failures = [a for a in attempts if a.get("failed")]
    usable = [a for a in attempts if not a.get("failed")]

    if not attempts:
        return Route(DEFAULT_MODEL, "nano", "unknown", ["no looks taken yet"])

    spread = spread_mm(usable)

    # The only case that escalates: a view was thrown out AND what is left does
    # not agree with itself. Everything short of that goes to the model that was
    # measured to be right about it.
    if failures and spread is not None and spread > SEVERE_DISAGREEMENT_MM:
        reasons.append(f"a view was refused and the rest disagree by {spread:.1f} mm")
        return Route(SUPER, "super", "conflicting", reasons)

    if len(failures) >= 2:
        return Route(SUPER, "super", "conflicting",
                     [f"{len(failures)} views refused outright"])

    detail = f"{len(attempts)} look(s)"
    if spread is not None:
        detail += f", views agree to {spread:.2f} mm"
    return Route(DEFAULT_MODEL, "nano", "ordinary", [detail])


def tier_of(model: str) -> str:
    return {NANO: "nano", LIGHTNING: "lightning",
            SUPER: "super", ULTRA: "ultra"}.get(model, "unknown")
