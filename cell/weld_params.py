"""Choosing the weld to suit the joint that is actually there.

Tracking the seam tells the torch where to go. It does not tell it how to weld,
and on a folded joint those are different questions, because the gap between the
two edges is not constant. A schedule that lays a sound weld on a tight fit will
blow straight through a gap twice as wide, and a schedule safe enough for the
wide part will leave the tight part cold and under-fused.

The adjustment is not the obvious one. The instinct on seeing a wider gap is to
turn the power up, and that is backwards: more arc force on an open joint pushes
the puddle through it. Practice is the reverse. Current and voltage come DOWN so
the arc is gentler, and the extra metal needed to fill the gap is bought by
slowing the torch, because deposition per millimetre is governed by travel speed.

Slowing down has a price, and it is the price this cell has to weigh. Heat input
is power over speed, so every millimetre per second surrendered to fill a gap
raises the heat going into the part, and heat is what distorts it. Filling the
joint better and moving the joint less are in direct opposition, and something has
to arbitrate.

Plain arithmetic, no simulation dependency, nothing random.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Starting amperage for mild steel follows a shop rule: roughly thirty-five to
# forty amps per millimetre of thickness for a single pass in the flat position.
# A corner joint on folded sheet sits at the lower end, because there is nowhere
# for the heat to go and thin material punishes an over-hot arc.
AMPS_PER_MM = 40.0

# Short-circuit GMAW on thin sheet. Voltage tracks current loosely; this is the
# working point rather than a law.
BASE_VOLTAGE_V = 19.0
BASE_TRAVEL_MM_S = 10.0

# What the joint will tolerate, as a multiple of sheet thickness. Up to about
# eight tenths of the thickness is an ordinary fit that needs no comment. Past
# about one and a half thicknesses the joint cannot be bridged in a single pass
# without backing, and the honest answer is to stop rather than to lay metal into
# a hole and call it a weld.
COMFORTABLE_GAP_PER_THICKNESS = 0.8
MAX_GAP_PER_THICKNESS = 1.6

# Heat input window for thin sheet, in joules per millimetre. Below the floor the
# arc has not put in enough energy to fuse the root. Above the ceiling a 2 mm
# sheet burns through.
HEAT_FLOOR_J_PER_MM = 95.0
HEAT_CEILING_PER_MM_THICKNESS = 140.0

# How far the arc is backed off once the joint is fully open. Saturates: past a
# point a softer arc stops helping and starts failing to fuse.
CURRENT_BACKOFF = 0.18

# CALIBRATING THE GAP READING, and what it is not yet good enough for.
#
# The tracker reports how far short of the fitted corner each edge's material
# stops. On this joint that is never zero even when the edges are touching,
# because the corner notch leaves an opening of about one sheet thickness on each
# side by construction. So a raw reading has to be referenced against what a
# known-good joint reads, which is an ordinary sensor calibration.
#
# Measured over fifty parts, none of which had any gap fault injected, the reading
# came back with a median of 10.5 mm. That IS what a touching joint reads on this
# geometry, so it is the reference, and the excess is what a fault would add on top
# of it. Subtracting the nominal notch as well was wrong and took it off twice.
#
# The offset is systematic and can be subtracted. The part-to-part spread cannot:
# it is 1.1 mm, and it is the honest uncertainty on this measurement.
#
# That uncertainty is larger than the margin between a gap this joint takes without
# comment and one it cannot bridge at all. So the schedule below is ADVISORY on
# this evidence: it will tell an operator the joint is opening up, and it must not
# be the thing that scraps a part until the reading is tightened. Saying otherwise
# would be dressing a 1 mm uncertainty up as a 0.1 mm decision.
GAP_READING_OFFSET_MM = 10.5
GAP_READING_UNCERTAINTY_MM = 1.1


def excess_gap_mm(measured_gap_mm: float) -> float:
    """Gap beyond what a touching joint of this geometry reads.

    Never negative: a reading under the reference means the joint is closed, not
    that it has negative width.
    """
    return max(0.0, measured_gap_mm - GAP_READING_OFFSET_MM)


# A fillet on sheet is laid with a leg about the thickness of the material. The
# cross-section the torch has to fill is that leg plus whatever the gap adds, and
# it is the fill area, not the gap, that sets how fast the torch can move.
NOMINAL_LEG_PER_THICKNESS = 1.0


@dataclass(frozen=True)
class Schedule:
    """One set of torch settings, and why they were chosen."""

    current_a: float
    voltage_v: float
    travel_mm_s: float
    gap_mm: float
    heat_input_j_per_mm: float
    weldable: bool
    reason: str
    needs_attention: bool = False

    def describe(self) -> str:
        if not self.weldable:
            return f"refused: {self.reason}"
        flag = "  [flagged]" if self.needs_attention else ""
        return (f"{self.current_a:.0f} A, {self.voltage_v:.1f} V, "
                f"{self.travel_mm_s:.1f} mm/s, {self.heat_input_j_per_mm:.0f} J/mm"
                f"{flag}")


@dataclass(frozen=True)
class ProcessLimits:
    """What this joint on this material will and will not take."""

    thickness_mm: float = 2.0
    efficiency: float = 0.80

    def __post_init__(self) -> None:
        if self.thickness_mm <= 0:
            raise ValueError("thickness must be positive")

    @property
    def comfortable_gap_mm(self) -> float:
        return COMFORTABLE_GAP_PER_THICKNESS * self.thickness_mm

    @property
    def max_gap_mm(self) -> float:
        return MAX_GAP_PER_THICKNESS * self.thickness_mm

    @property
    def heat_ceiling_j_per_mm(self) -> float:
        return HEAT_CEILING_PER_MM_THICKNESS * self.thickness_mm

    def base_current_a(self) -> float:
        return AMPS_PER_MM * self.thickness_mm


def heat_input_j_per_mm(current_a: float, voltage_v: float, travel_mm_s: float,
                        efficiency: float = 0.80) -> float:
    if travel_mm_s <= 0:
        raise ValueError("travel speed must be positive")
    return efficiency * current_a * voltage_v / travel_mm_s


def schedule_for(gap_mm: float, limits: Optional[ProcessLimits] = None) -> Schedule:
    """Pick torch settings for a measured gap.

    Returns a refusal rather than a bad weld when the joint is beyond what one
    pass can bridge. A cell that never refuses anything is not measuring, it is
    decorating.
    """
    limits = limits or ProcessLimits()
    if gap_mm < 0:
        raise ValueError("gap must not be negative")

    base_current = limits.base_current_a()

    if gap_mm > limits.max_gap_mm:
        return Schedule(
            current_a=0.0, voltage_v=0.0, travel_mm_s=0.0, gap_mm=gap_mm,
            heat_input_j_per_mm=0.0, weldable=False,
            reason=(f"gap {gap_mm:.2f} mm is past {limits.max_gap_mm:.2f} mm, which is "
                    f"{MAX_GAP_PER_THICKNESS:g} times the {limits.thickness_mm:.1f} mm "
                    "sheet. One pass cannot bridge it without backing; the press "
                    "fit-up needs correcting rather than the weld"),
        )

    # How far open the joint is, against what it comfortably takes. The arc backs
    # off as it opens and then stops backing off, because a soft arc that cannot
    # fuse the root is no better than a hard one that blows through it.
    openness = min(1.0, gap_mm / limits.comfortable_gap_mm) if limits.comfortable_gap_mm else 0.0

    # Down, not up. A gentler arc is less likely to push the puddle through an
    # open root.
    current = base_current * (1.0 - CURRENT_BACKOFF * openness)
    voltage = BASE_VOLTAGE_V * (1.0 - 0.5 * CURRENT_BACKOFF * openness)

    # The extra metal comes from time, not from power, and how much extra is set
    # by the cross-section to be filled. A gap twice as wide is not twice as slow;
    # it is slower in proportion to the area it adds to the fillet already being
    # laid. Scaling off the gap alone saturates and leaves a 3 mm gap running the
    # same schedule as a 1.6 mm one.
    leg = NOMINAL_LEG_PER_THICKNESS * limits.thickness_mm
    nominal_area = limits.thickness_mm * leg
    fill_area = limits.thickness_mm * (leg + gap_mm)
    travel = BASE_TRAVEL_MM_S * nominal_area / fill_area

    heat = heat_input_j_per_mm(current, voltage, travel, limits.efficiency)

    # Slowing down to fill is only free until the sheet starts burning through.
    if heat > limits.heat_ceiling_j_per_mm:
        travel = limits.efficiency * current * voltage / limits.heat_ceiling_j_per_mm
        heat = heat_input_j_per_mm(current, voltage, travel, limits.efficiency)
        reason = (f"gap {gap_mm:.2f} mm wants more fill than the sheet will take; "
                  f"travel held at {travel:.1f} mm/s to keep heat input under "
                  f"{limits.heat_ceiling_j_per_mm:.0f} J/mm. Fill may be short")
        return Schedule(current, voltage, travel, gap_mm, heat, True, reason,
                        needs_attention=True)

    if heat < HEAT_FLOOR_J_PER_MM:
        travel = limits.efficiency * current * voltage / HEAT_FLOOR_J_PER_MM
        heat = heat_input_j_per_mm(current, voltage, travel, limits.efficiency)
        reason = (f"heat input would have fallen under {HEAT_FLOOR_J_PER_MM:.0f} J/mm "
                  f"and left the root unfused; travel slowed to {travel:.1f} mm/s")
        return Schedule(current, voltage, travel, gap_mm, heat, True, reason,
                        needs_attention=True)

    if gap_mm <= limits.comfortable_gap_mm:
        reason = (f"gap {gap_mm:.2f} mm is inside the {limits.comfortable_gap_mm:.2f} mm "
                  "the joint takes without comment")
    else:
        reason = (f"gap {gap_mm:.2f} mm is past the comfortable "
                  f"{limits.comfortable_gap_mm:.2f} mm; arc backed off and travel slowed "
                  "to fill it, which puts more heat into the part")

    return Schedule(current, voltage, travel, gap_mm, heat, True, reason,
                    needs_attention=gap_mm > limits.comfortable_gap_mm)


def schedule_along_seam(gaps_mm, limits: Optional[ProcessLimits] = None) -> list[Schedule]:
    """A schedule per station, because the gap is not the same all the way up.

    A single set of parameters for the whole seam is a compromise against a joint
    that changes along its length. This is what the tracking measurement is for:
    having measured the gap at every station, the torch can be told what to do at
    every station.
    """
    return [schedule_for(gap, limits) for gap in gaps_mm]


def summarise(schedules: list[Schedule]) -> dict:
    """What the run as a whole looks like, for a log or an operator."""
    weldable = [s for s in schedules if s.weldable]
    return {
        "stations": len(schedules),
        "weldable": len(weldable),
        "refused": len(schedules) - len(weldable),
        "flagged": sum(1 for s in weldable if s.needs_attention),
        "gap_mm_min": min((s.gap_mm for s in schedules), default=0.0),
        "gap_mm_max": max((s.gap_mm for s in schedules), default=0.0),
        "travel_mm_s_min": min((s.travel_mm_s for s in weldable), default=0.0),
        "travel_mm_s_max": max((s.travel_mm_s for s in weldable), default=0.0),
        "heat_j_per_mm_min": min((s.heat_input_j_per_mm for s in weldable), default=0.0),
        "heat_j_per_mm_max": max((s.heat_input_j_per_mm for s in weldable), default=0.0),
    }
