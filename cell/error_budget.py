"""Everything between the joint and the torch, added up honestly.

The seam study says the tracker finds the joint to 0.67 mm. That is a true number
about one link in a chain, and quoting it as the accuracy of the cell would be a
misrepresentation. Published work on vision guided welding is explicit that the
achieved accuracy depends on the whole chain: sensor resolution, calibration
quality, standoff stability, part surface, robot repeatability and the arc
environment. This repository measures two of those terms carefully and had been
silent about the rest.

The silence mattered most for one of them. Hand eye calibration, which relates
what the camera sees to where the robot goes, carries a residual of roughly 0.7 to
1.0 mm after a good calibration on an industrial arm, and several millimetres
before one. That is the same size as the entire tracking error, so leaving it out
does not make the budget slightly optimistic, it halves it.

Terms are combined in quadrature where they are independent random errors and
added where they are systematic, which is the distinction that decides whether a
term matters. A systematic offset the same on every part is not averaged away by
measuring more parts, and several of these are systematic.

What this module does NOT do is claim the unmeasured terms were measured. Each one
carries where its number came from, and the ones taken from published ranges say
so. A budget whose provenance is not visible is a budget nobody can argue with,
which is the opposite of useful. The first version of this file got that wrong in
its own laser row, labelling a published figure as measured here, which is a good
illustration of how easily it happens.

One term connects to the rest of the cell. Hand eye calibration is both the largest
unmeasured contribution and, on the seam alone, undetectable: a camera whose
calibration has crept moves every measurement equally at the bottom and the top,
which is precisely the signature of a fixture creeping. The reference feature added
for the drift diagnosis is the same instrument that would catch it. That is not a
coincidence, it is the same blind spot seen from two directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Kind = Literal["random", "systematic"]
Source = Literal["measured here", "published range", "specification"]


@dataclass(frozen=True)
class Term:
    """One contribution to the distance between the torch and the joint."""

    name: str
    mm: float
    kind: Kind
    source: Source
    fixable: bool
    note: str
    cite: str = ""

    @property
    def averages_down(self) -> bool:
        """Whether making more parts helps. Systematic terms do not."""
        return self.kind == "random"


@dataclass
class Budget:
    """A named chain of terms, and what it comes to."""

    label: str
    terms: list[Term] = field(default_factory=list)

    def random_mm(self) -> float:
        """Independent errors combine in quadrature."""
        return math.sqrt(sum(t.mm ** 2 for t in self.terms if t.kind == "random"))

    def systematic_mm(self) -> float:
        """Offsets that are the same on every part simply add."""
        return sum(t.mm for t in self.terms if t.kind == "systematic")

    def total_mm(self) -> float:
        return self.systematic_mm() + self.random_mm()

    def dominant(self) -> Term | None:
        return max(self.terms, key=lambda t: t.mm, default=None)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "terms": [t.__dict__ | {"averages_down": t.averages_down} for t in self.terms],
            "random_mm": self.random_mm(),
            "systematic_mm": self.systematic_mm(),
            "total_mm": self.total_mm(),
            "dominant": self.dominant().name if self.terms else None,
        }


# What the cell measured about itself. These are the only numbers here that came
# out of a run in this repository, and they are marked as such.
TRACKING_CLEAN_MM = 0.676        # seam campaign, fifty parts, exact depth
TRACKING_STEREO_MM = 1.787       # the same campaign through a typical stereo sensor
SEAM_BOW_MM = 0.39              # the seam's own curvature, mean over fifty parts
THERMAL_UNCOMPENSATED_MM = 0.552  # where the arc leaves the joint, uncompensated
THERMAL_COMPENSATED_MM = 0.034   # the residual after compensating at nominal

# What the cell did not measure, taken from published figures for industrial arms
# and vision guidance. Ranges rather than points, and the pessimistic end is used,
# because a budget built on the optimistic end of everything is a wish.
HAND_EYE_MM = 0.85              # typical residual after calibration, 0.7 to 1.0
ROBOT_REPEATABILITY_MM = 0.08    # 0.05 to 0.1 on a welding class arm
STANDOFF_DRIFT_MM = 0.15        # thermal growth of the mount over a shift

# What industrial vision guidance for weld tracking is asked to achieve. Quoted so
# the budget is read against a requirement rather than against nothing.
INDUSTRY_REQUIREMENT_MM = 1.5

# What a laser line profiler achieves, which is the class of sensor the robustness
# sweep says this method needs.
LASER_PROFILER_MM = 0.1


def _shared(tracking_mm: float, tracking_note: str, thermal_mm: float,
            thermal_note: str,
            tracking_source: Source = "measured here",
            tracking_cite: str = "seam campaign, 50 parts x 20 measured points") -> list[Term]:
    return [
        Term("Seam tracking", tracking_mm, "random", tracking_source, True,
             tracking_note, tracking_cite),
        Term("Hand eye calibration", HAND_EYE_MM, "systematic", "published range", False,
             "Relates what the camera sees to where the arm goes. The same offset on "
             "every part, so measuring more parts does not average it away, and this "
             "cell has no way to detect it drifting without a reference feature.",
             "typically 0.7 to 1.0 mm left after calibration on an industrial arm"),
        Term("Robot repeatability", ROBOT_REPEATABILITY_MM, "random", "specification", False,
             "What the arm does when told to go to the same point twice. Small enough "
             "on a welding class machine to be the least of the worries here.",
             "0.05 to 0.1 mm, manufacturer specification"),
        Term("Standoff drift", STANDOFF_DRIFT_MM, "systematic", "published range", False,
             "The camera mount growing as the cell warms through a shift. Slow, and "
             "indistinguishable from calibration drift in the seam alone.", ""),
        Term("Thermal distortion", thermal_mm, "systematic", "measured here", True,
             thermal_note, "compensation study, 120 trials per level"),
    ]


def budgets() -> list[Budget]:
    """The chain under four sets of assumptions, from flattering to realistic."""
    return [
        Budget("What the seam study alone would suggest", [
            Term("Seam tracking", TRACKING_CLEAN_MM, "random", "measured here", True,
                 "The number this repository quotes most often, and the only term in "
                 "this column. Correct about the tracker and incomplete about the cell.",
                 "seam campaign, 50 parts x 20 measured points"),
        ]),
        Budget("Perfect sensor, nothing compensated", _shared(
            TRACKING_CLEAN_MM,
            "On exact depth, which no sensor delivers. Held here to isolate what the "
            "other terms cost.",
            THERMAL_UNCOMPENSATED_MM,
            "The arc pulls the joint inward as it welds and the path was driven to the "
            "cold measurement.")),
        Budget("Typical stereo camera, compensated", _shared(
            TRACKING_STEREO_MM,
            "The same campaign through a stereo sensor at a metre: 2.5 mm range noise, "
            "10 percent of the frame lost to the shiny sheet, false points at the joint edge.",
            THERMAL_COMPENSATED_MM,
            "Compensated at nominal, welding where the joint will be rather than where "
            "it was.")),
        Budget("Laser profiler, compensated", _shared(
            LASER_PROFILER_MM,
            "The sensor class the robustness sweep says this method needs, and the one "
            "production seam trackers use. This row is what the chain would come to "
            "with one, not something this repository ran.",
            THERMAL_COMPENSATED_MM,
            "Compensated at nominal.",
            tracking_source="published range",
            tracking_cite="commercial and published laser seam trackers report "
                          "0.14 to 0.25 mm tracking error")),
    ]


def summary() -> dict:
    out = [b.as_dict() for b in budgets()]
    realistic = next(b for b in out if b["label"].startswith("Typical stereo"))
    best = next(b for b in out if b["label"].startswith("Laser"))
    return {
        "budgets": out,
        "industry_requirement_mm": INDUSTRY_REQUIREMENT_MM,
        "realistic_total_mm": realistic["total_mm"],
        "best_total_mm": best["total_mm"],
        "meets_requirement": best["total_mm"] < INDUSTRY_REQUIREMENT_MM,
        "dominant_unmeasured": "Hand eye calibration",
    }
