"""Weld defect injection.

Produces a plan describing what the bead should look like for one part, together
with the label of whatever was deliberately wrong with it. The simulation builds
geometry from the plan; the label is the ground truth the cloud inspector is
scored against.

No Isaac or numpy dependency, so the plans can be generated, inspected and tested
anywhere. Seeded, so a campaign replays identically.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

# Kept as plain strings to match the service enum without importing it, since this
# module is also imported by the Isaac interpreter.
NONE = "none"
UNDERCUT = "undercut"
SKIP = "skip"
POROSITY = "porosity"
SPATTER = "spatter"
BURN_THROUGH = "burn_through"

DEFECT_KINDS = (UNDERCUT, SKIP, POROSITY, SPATTER, BURN_THROUGH)


@dataclass
class Blob:
    """A small feature on or beside the bead, positioned along it.

    along: 0 at the bottom of the joint, 1 at the top.
    offset: lateral displacement from the bead centreline, in metres.
    radius: in metres.
    """

    along: float
    offset: float
    radius: float
    kind: str


@dataclass
class BeadPlan:
    """How to build one bead, and what is wrong with it."""

    defect: str = NONE
    severity: float = 0.0
    # Spans of the joint that actually carry weld metal, as (start, end) in 0..1.
    # A continuous bead is a single span covering the whole joint.
    segments: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.0)])
    blobs: list[Blob] = field(default_factory=list)
    # Undercut groove: (start, end, depth_m) along the bead toe.
    groove: Optional[tuple[float, float, float]] = None
    # Burn-through hole: (along, radius_m).
    hole: Optional[tuple[float, float]] = None

    @property
    def clean(self) -> bool:
        return self.defect == NONE

    def describe(self) -> str:
        """One line for logs. Never shown to the inspection model."""
        if self.clean:
            return "clean bead"
        return f"{self.defect} (severity {self.severity:.2f})"


def clean_bead() -> BeadPlan:
    return BeadPlan()


def _skip(rng: random.Random) -> BeadPlan:
    """Bead stops and restarts, leaving bare joint."""
    gap_len = rng.uniform(0.08, 0.22)
    start = rng.uniform(0.15, 0.85 - gap_len)
    return BeadPlan(
        defect=SKIP,
        severity=min(1.0, gap_len / 0.22),
        segments=[(0.0, start), (start + gap_len, 1.0)],
    )


def _porosity(rng: random.Random) -> BeadPlan:
    """Gas pores in the bead face."""
    count = rng.randint(3, 9)
    return BeadPlan(
        defect=POROSITY,
        severity=min(1.0, count / 9.0),
        blobs=[
            Blob(
                along=rng.uniform(0.1, 0.9),
                offset=rng.uniform(-0.004, 0.004),
                radius=rng.uniform(0.0015, 0.0035),
                kind=POROSITY,
            )
            for _ in range(count)
        ],
    )


def _spatter(rng: random.Random) -> BeadPlan:
    """Loose droplets thrown onto the parent metal beside the bead."""
    count = rng.randint(5, 14)
    return BeadPlan(
        defect=SPATTER,
        severity=min(1.0, count / 14.0),
        blobs=[
            Blob(
                along=rng.uniform(0.05, 0.95),
                offset=rng.choice([-1, 1]) * rng.uniform(0.018, 0.045),
                radius=rng.uniform(0.0012, 0.0030),
                kind=SPATTER,
            )
            for _ in range(count)
        ],
    )


def _undercut(rng: random.Random) -> BeadPlan:
    """A groove melted into the parent metal along the bead toe."""
    length = rng.uniform(0.2, 0.55)
    start = rng.uniform(0.05, 0.95 - length)
    depth = rng.uniform(0.0008, 0.0025)
    return BeadPlan(
        defect=UNDERCUT,
        severity=min(1.0, depth / 0.0025),
        groove=(start, start + length, depth),
    )


def _burn_through(rng: random.Random) -> BeadPlan:
    """A hole melted fully through the joint."""
    radius = rng.uniform(0.004, 0.009)
    return BeadPlan(
        defect=BURN_THROUGH,
        severity=min(1.0, radius / 0.009),
        hole=(rng.uniform(0.2, 0.8), radius),
    )


_BUILDERS = {
    SKIP: _skip,
    POROSITY: _porosity,
    SPATTER: _spatter,
    UNDERCUT: _undercut,
    BURN_THROUGH: _burn_through,
}


def plan_bead(
    rng: random.Random,
    defect_rate: float = 0.35,
    kinds: tuple[str, ...] = DEFECT_KINDS,
) -> BeadPlan:
    """Decide what this part's bead looks like.

    defect_rate is the share of parts carrying a defect. A campaign wants enough
    clean parts to measure false alarms and enough defective ones to measure
    misses, so the default is well above a real line's scrap rate on purpose.
    """
    if not 0.0 <= defect_rate <= 1.0:
        raise ValueError("defect_rate must be between 0 and 1")
    if rng.random() >= defect_rate:
        return clean_bead()
    return _BUILDERS[rng.choice(list(kinds))](rng)


def plan_campaign(
    n: int,
    seed: int = 0,
    defect_rate: float = 0.35,
    kinds: tuple[str, ...] = DEFECT_KINDS,
) -> list[BeadPlan]:
    """A reproducible sequence of bead plans for a scoring campaign."""
    rng = random.Random(seed)
    return [plan_bead(rng, defect_rate=defect_rate, kinds=kinds) for _ in range(n)]
