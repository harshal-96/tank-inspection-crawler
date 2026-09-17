"""Tests for choosing the weld to suit the joint that is there.

The checks are against welding practice, not against whatever the function
returns. The one that matters most is that the arc is turned DOWN as the joint
opens, because the intuitive answer is to turn it up and that is how a puddle
gets pushed through an open root.
"""

from __future__ import annotations

import pytest

from cell.weld_params import (
    HEAT_FLOOR_J_PER_MM,
    ProcessLimits,
    heat_input_j_per_mm,
    schedule_along_seam,
    schedule_for,
    summarise,
)

SHEET = ProcessLimits(thickness_mm=2.0)


# ------------------------------------------------------------------ the basics


def test_amperage_follows_the_thickness_rule():
    """Roughly forty amps per millimetre for mild steel."""
    assert ProcessLimits(thickness_mm=2.0).base_current_a() == pytest.approx(80.0)
    assert ProcessLimits(thickness_mm=3.0).base_current_a() == pytest.approx(120.0)


def test_heat_input_is_power_over_speed():
    assert heat_input_j_per_mm(80, 19, 10, 0.8) == pytest.approx(0.8 * 80 * 19 / 10)


def test_a_stopped_torch_is_refused():
    with pytest.raises(ValueError):
        heat_input_j_per_mm(80, 19, 0.0)


def test_a_negative_gap_is_refused():
    with pytest.raises(ValueError):
        schedule_for(-0.1, SHEET)


def test_zero_thickness_is_refused():
    with pytest.raises(ValueError):
        ProcessLimits(thickness_mm=0.0)


# ------------------------------------------- the counterintuitive bit


def test_the_arc_is_turned_DOWN_as_the_joint_opens():
    """The whole point. More arc force on an open root pushes the puddle through."""
    tight = schedule_for(0.0, SHEET)
    open_joint = schedule_for(1.6, SHEET)
    assert open_joint.current_a < tight.current_a
    assert open_joint.voltage_v < tight.voltage_v


def test_the_fill_comes_from_slowing_down_not_from_power():
    tight = schedule_for(0.0, SHEET)
    open_joint = schedule_for(1.6, SHEET)
    assert open_joint.travel_mm_s < tight.travel_mm_s * 0.7


def test_filling_a_wider_gap_costs_heat():
    """The trade the cell has to arbitrate: fill better, distort more."""
    tight = schedule_for(0.0, SHEET)
    open_joint = schedule_for(1.6, SHEET)
    assert open_joint.heat_input_j_per_mm > tight.heat_input_j_per_mm


def test_travel_keeps_falling_past_the_comfortable_gap():
    """A three millimetre gap needs more fill than a 1.6 mm one.

    Scaling the response off the gap alone saturates at the comfortable limit and
    runs a wide joint on the same schedule as a snug one.
    """
    speeds = [schedule_for(g, SHEET).travel_mm_s for g in (1.6, 2.2, 3.0, 3.2)]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] < speeds[0] * 0.8


def test_heat_rises_monotonically_with_the_gap():
    heats = [schedule_for(g, SHEET).heat_input_j_per_mm for g in (0.0, 0.8, 1.6, 2.4, 3.2)]
    assert heats == sorted(heats)


# --------------------------------------------------------------- the limits


def test_a_gap_past_what_one_pass_can_bridge_is_refused():
    beyond = schedule_for(SHEET.max_gap_mm + 0.3, SHEET)
    assert not beyond.weldable
    assert beyond.travel_mm_s == 0.0


def test_the_refusal_blames_the_press_not_the_weld():
    """The useful answer on an unbridgeable gap is to fix the fit-up."""
    beyond = schedule_for(SHEET.max_gap_mm + 0.3, SHEET)
    assert "fit-up" in beyond.reason


def test_the_widest_bridgeable_gap_is_still_accepted():
    edge = schedule_for(SHEET.max_gap_mm, SHEET)
    assert edge.weldable
    assert edge.needs_attention


def test_a_comfortable_gap_is_not_flagged():
    assert not schedule_for(0.4, SHEET).needs_attention


def test_heat_never_exceeds_the_burn_through_ceiling():
    for gap in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.2):
        schedule = schedule_for(gap, SHEET)
        if schedule.weldable:
            assert schedule.heat_input_j_per_mm <= SHEET.heat_ceiling_j_per_mm + 1e-6


def test_heat_never_falls_below_the_fusion_floor():
    for gap in (0.0, 0.5, 1.0, 2.0, 3.0):
        schedule = schedule_for(gap, SHEET)
        if schedule.weldable:
            assert schedule.heat_input_j_per_mm >= HEAT_FLOOR_J_PER_MM - 1e-6


def test_thicker_material_takes_a_wider_gap_and_more_current():
    thin, thick = ProcessLimits(thickness_mm=2.0), ProcessLimits(thickness_mm=4.0)
    assert thick.max_gap_mm > thin.max_gap_mm
    assert schedule_for(0.0, thick).current_a > schedule_for(0.0, thin).current_a


def test_a_gap_that_stops_a_thin_sheet_is_fine_on_a_thick_one():
    gap = 2.5
    assert not schedule_for(gap, ProcessLimits(thickness_mm=1.2)).weldable
    assert schedule_for(gap, ProcessLimits(thickness_mm=4.0)).weldable


# -------------------------------------------------------------- along the seam


def test_every_station_gets_its_own_schedule():
    """A single schedule for the whole seam is a compromise against a joint that
    changes along its length. Having measured it station by station, there is no
    reason to accept that compromise."""
    gaps = [0.1, 0.4, 0.9, 1.5, 2.1]
    schedules = schedule_along_seam(gaps, SHEET)
    assert len(schedules) == len(gaps)
    speeds = [s.travel_mm_s for s in schedules]
    assert speeds == sorted(speeds, reverse=True)


def test_a_seam_with_one_unbridgeable_station_is_reported_as_such():
    schedules = schedule_along_seam([0.2, 0.3, 4.0, 0.3], SHEET)
    report = summarise(schedules)
    assert report["refused"] == 1
    assert report["weldable"] == 3
    assert report["gap_mm_max"] == pytest.approx(4.0)


def test_the_summary_survives_a_seam_where_nothing_is_weldable():
    report = summarise(schedule_along_seam([9.0, 9.0], SHEET))
    assert report["weldable"] == 0
    assert report["travel_mm_s_min"] == 0.0
