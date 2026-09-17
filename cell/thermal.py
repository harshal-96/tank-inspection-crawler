"""What the arc does to the joint while it is welding it.

Everything measured before the arc strikes describes a cold part. The moment the
arc starts, the part stops being the part that was measured. Weld metal and the
heat-affected zone contract as they cool and pull the two sides together, so the
joint closes up behind the arc and the corner walks inward. Ahead of the arc the
metal is merely warm and has expanded slightly, which opens the gap a little
before it closes a lot.

This is not a small correction bolted onto a large one. The press leaves around
two tenths of a degree of residual springback; the literature puts angular
distortion from a weld at about a quarter of a degree for a run of this size, and
transverse shrinkage near a millimetre over half a metre of seam. So the arc moves
the joint by as much as the press did, and it does it after the measurement was
taken. A cell that photographs a cold part, drives the path it found, and never
looks again is welding a part that no longer exists.

Angular distortion rises with heat input and falls with the square of the
thickness, which is why thin sheet is the hard case: the same weld that barely
troubles a plate folds a 2 mm side over visibly.

Plain arithmetic, no simulation dependency. Seeded where it is random, so a
campaign replays identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Arc efficiency for gas metal arc welding: the share of electrical power that
# actually reaches the work. The rest goes to radiation and spatter.
GMAW_EFFICIENCY = 0.80

# Calibration. Both constants are set so that a run at roughly 190 J/mm on 2 mm
# sheet reproduces the figures reported for welds of this size: about a quarter of
# a degree of angular distortion and about nine tenths of a millimetre of
# transverse shrinkage over half a metre. They are not free parameters; changing
# them changes what this claims about real welds.
ANGULAR_DEG_PER_KJ_PER_MM = 5.3     # divided by thickness squared, in mm
SHRINK_MM_PER_KJ_PER_MM = 9.9       # divided by thickness, in mm

# How far behind the arc the shrinkage needs to develop. Weld metal does not
# finish contracting the instant it solidifies; it keeps pulling as it cools
# through to ambient, over a stretch of seam behind the torch.
COOLING_LENGTH_MM = 70.0

# How far ahead of the arc the parent metal is meaningfully warmed by conduction.
PREHEAT_REACH_MM = 40.0

# Warm metal ahead of the arc has expanded, which opens the joint slightly before
# the shrinkage behind closes it. Small, but it is the reason a gap read just
# ahead of the torch is not the gap the torch will meet.
PREHEAT_OPENING_FRACTION = 0.18


@dataclass(frozen=True)
class WeldPass:
    """The parameters the torch is actually running.

    These are what an adaptive cell changes when the joint changes, so they are
    inputs here rather than constants: the distortion a weld causes depends on
    the weld you chose to make.
    """

    current_a: float = 120.0
    voltage_v: float = 20.0
    travel_mm_s: float = 10.0
    efficiency: float = GMAW_EFFICIENCY

    def __post_init__(self) -> None:
        if self.travel_mm_s <= 0:
            raise ValueError("travel speed must be positive")
        if self.current_a < 0 or self.voltage_v < 0:
            raise ValueError("current and voltage must not be negative")

    def heat_input_j_per_mm(self) -> float:
        """Energy delivered per millimetre of seam.

        The single number that governs how much the part moves. Slowing the torch
        down to fill a wider gap raises it, which is the trade an adaptive cell is
        making whether it knows it or not.
        """
        return self.efficiency * self.current_a * self.voltage_v / self.travel_mm_s

    def heat_input_kj_per_mm(self) -> float:
        return self.heat_input_j_per_mm() / 1000.0


@dataclass(frozen=True)
class ThermalDistortion:
    """How the joint moves as the arc travels along it.

    Positions are millimetres along the seam from where the weld started. The arc
    position is where the torch is now, so the same seam reads differently at
    every instant of the weld, which is the whole problem.
    """

    weld: WeldPass = WeldPass()
    thickness_mm: float = 2.0
    seam_length_mm: float = 450.0

    def __post_init__(self) -> None:
        if self.thickness_mm <= 0 or self.seam_length_mm <= 0:
            raise ValueError("thickness and seam length must be positive")

    # ------------------------------------------------------------------ totals

    def full_shrink_mm(self) -> float:
        """Transverse shrinkage once a stretch of seam has fully cooled."""
        return (SHRINK_MM_PER_KJ_PER_MM * self.weld.heat_input_kj_per_mm()
                / self.thickness_mm)

    def full_angular_deg(self) -> float:
        """Angular distortion once the whole seam has been welded and cooled."""
        return (ANGULAR_DEG_PER_KJ_PER_MM * self.weld.heat_input_kj_per_mm()
                / (self.thickness_mm ** 2))

    # ------------------------------------------------------------ along the seam

    def _developed(self, along_mm: float, arc_mm: float) -> float:
        """How much of the shrinkage at a point has happened yet, 0 to 1.

        Zero ahead of the arc, rising behind it as the metal cools, saturating
        once it is cold.
        """
        behind = arc_mm - along_mm
        if behind <= 0.0:
            return 0.0
        return min(1.0, behind / COOLING_LENGTH_MM)

    def _preheat(self, along_mm: float, arc_mm: float) -> float:
        """How warm the metal at a point is from the approaching arc, 0 to 1."""
        ahead = along_mm - arc_mm
        if ahead < 0.0:
            return 0.0
        return math.exp(-ahead / PREHEAT_REACH_MM)

    def cooled_length_mm(self, arc_mm: float) -> float:
        """How much seam has been welded AND has finished pulling.

        Not simply how far the torch has travelled. Weld metal keeps contracting
        as it cools through to ambient, so the stretch immediately behind the arc
        has only partly done its pulling. This integrates how far each millimetre
        of welded seam has got with its shrinkage.
        """
        arc = max(0.0, min(arc_mm, self.seam_length_mm))
        if arc <= COOLING_LENGTH_MM:
            return arc * arc / (2.0 * COOLING_LENGTH_MM)
        return COOLING_LENGTH_MM / 2.0 + (arc - COOLING_LENGTH_MM)

    def rotation_deg(self, arc_mm: float) -> float:
        """How far the sides have rotated about their folds so far.

        Shrinkage along the seam pulls the two sides together, and they hinge
        about the fold lines that hold them to the base. The angle grows as more
        of the seam cools, so it is a function of where the torch has got to.
        """
        fraction = self.cooled_length_mm(arc_mm) / self.seam_length_mm
        return self.full_angular_deg() * fraction

    def seam_shift_mm(self, along_mm: float, arc_mm: float) -> float:
        """How far the seam at a given height has walked inward, right now.

        This is the correction that a cold measurement cannot contain. Note that
        it applies AHEAD of the arc as well as behind it: the sides rotate as a
        whole, so welding the bottom of a seam moves the top of it, long before
        the torch gets there. A cell that measures a point and then drives to it
        is aiming at where that point used to be.
        """
        height = max(0.0, min(along_mm, self.seam_length_mm))
        return math.tan(math.radians(self.rotation_deg(arc_mm))) * height

    def gap_change_mm(self, along_mm: float, arc_mm: float) -> float:
        """How much the gap at a point has changed from its cold value.

        Negative closes the joint, positive opens it. Behind the arc the weld
        metal pulls the joint shut as it cools. Just ahead of it the parent metal
        has been warmed by conduction and expanded, so the joint is slightly wider
        than it was cold, which is why a gap read a few centimetres in front of
        the torch is not the gap the torch will meet.
        """
        closing = -self.full_shrink_mm() * self._developed(along_mm, arc_mm)
        opening = (self.full_shrink_mm() * PREHEAT_OPENING_FRACTION
                   * self._preheat(along_mm, arc_mm))
        return closing + opening


    # --------------------------------------------------------------- summaries

    def drift_at_finish_mm(self) -> float:
        """How far the top of the seam has walked by the time the weld is done."""
        return self.seam_shift_mm(self.seam_length_mm, self.seam_length_mm)

    def worst_drift_ahead_of_arc_mm(self, lead_mm: float, samples: int = 60) -> float:
        """The error a cell takes on if it measures ahead and never re-measures.

        A look-ahead sensor reads the joint a fixed distance in front of the
        torch. Between that reading and the torch arriving, more seam has cooled
        and the sides have rotated further, so the joint is no longer where it
        was read. This is how far it moves in that interval, at the worst point
        of the weld.

        It is the number that decides whether looking ahead once is enough or
        whether the cell has to keep looking.
        """
        worst = 0.0
        for i in range(samples + 1):
            arc = self.seam_length_mm * i / samples
            target = arc + lead_mm
            if target > self.seam_length_mm:
                break
            seen_when_measured = self.seam_shift_mm(target, arc)
            where_it_is_on_arrival = self.seam_shift_mm(target, target)
            worst = max(worst, abs(where_it_is_on_arrival - seen_when_measured))
        return worst

    def describe(self) -> str:
        return (f"{self.weld.heat_input_j_per_mm():.0f} J/mm, "
                f"shrink {self.full_shrink_mm():.2f} mm, "
                f"angular {self.full_angular_deg():.2f} deg, "
                f"corner walks {self.drift_at_finish_mm():.2f} mm")
