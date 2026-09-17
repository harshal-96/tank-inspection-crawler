"""Tests for what the arc does to the joint while it is welding it.

Checked against the physics rather than against whatever the code returns: heat
input rises when the torch slows, distortion rises with heat input and falls with
the square of the thickness, the joint closes behind the arc and opens slightly
in front of it, and welding the bottom of a seam moves the top of it.
"""

from __future__ import annotations

import pytest

from cell.thermal import (
    COOLING_LENGTH_MM,
    ThermalDistortion,
    WeldPass,
)

SEAM = 450.0


# ------------------------------------------------------------------ heat input


def test_heat_input_is_power_over_speed():
    weld = WeldPass(current_a=120.0, voltage_v=20.0, travel_mm_s=10.0, efficiency=0.8)
    assert weld.heat_input_j_per_mm() == pytest.approx(0.8 * 120 * 20 / 10)


def test_slowing_down_to_fill_a_gap_puts_in_more_heat():
    """The trade an adaptive cell makes whether it knows it or not."""
    fast = WeldPass(travel_mm_s=16.0).heat_input_j_per_mm()
    slow = WeldPass(travel_mm_s=6.0).heat_input_j_per_mm()
    assert slow > fast * 2.5


def test_a_stopped_torch_is_refused():
    with pytest.raises(ValueError):
        WeldPass(travel_mm_s=0.0)


def test_negative_current_is_refused():
    with pytest.raises(ValueError):
        WeldPass(current_a=-10.0)


# ------------------------------------------------------- magnitudes vs the trade


def test_reproduces_the_reported_distortion_for_a_weld_this_size():
    """Calibration check against published figures, not a snapshot of the code.

    A run near 190 J/mm on 2 mm sheet should land close to a quarter of a degree
    of angular distortion and about nine tenths of a millimetre of transverse
    shrinkage. If this test fails the model has stopped describing real welds.
    """
    hot = ThermalDistortion(thickness_mm=2.0, seam_length_mm=500.0)
    assert hot.weld.heat_input_j_per_mm() == pytest.approx(192.0, abs=1.0)
    assert hot.full_angular_deg() == pytest.approx(0.25, abs=0.05)
    assert hot.full_shrink_mm() == pytest.approx(0.94, abs=0.10)


def test_thin_sheet_is_the_hard_case():
    """Angular distortion falls with the SQUARE of thickness."""
    thin = ThermalDistortion(thickness_mm=2.0).full_angular_deg()
    thick = ThermalDistortion(thickness_mm=4.0).full_angular_deg()
    assert thin == pytest.approx(thick * 4.0, rel=0.01)


def test_more_heat_distorts_more():
    cool = ThermalDistortion(weld=WeldPass(travel_mm_s=16.0))
    hot = ThermalDistortion(weld=WeldPass(travel_mm_s=6.0))
    assert hot.full_angular_deg() > cool.full_angular_deg() * 2.0
    assert hot.full_shrink_mm() > cool.full_shrink_mm() * 2.0


def test_the_arc_moves_the_joint_as_much_as_the_press_did():
    """The reason a cold measurement is not enough.

    The press leaves a couple of tenths of a degree of residual springback. If
    the arc contributed far less than that it could be ignored; it does not.
    """
    from cell.forming import LEAN_SIGMA_DEG

    assert ThermalDistortion().full_angular_deg() > LEAN_SIGMA_DEG


def test_a_part_with_no_thickness_is_refused():
    with pytest.raises(ValueError):
        ThermalDistortion(thickness_mm=0.0)


# ------------------------------------------------------------- along the seam


def test_nothing_has_moved_before_the_arc_strikes():
    hot = ThermalDistortion(seam_length_mm=SEAM)
    assert hot.rotation_deg(0.0) == pytest.approx(0.0)
    assert hot.seam_shift_mm(SEAM, 0.0) == pytest.approx(0.0)


def test_the_joint_keeps_walking_as_the_weld_runs():
    hot = ThermalDistortion(seam_length_mm=SEAM)
    walk = [hot.seam_shift_mm(SEAM, arc) for arc in (0, 100, 200, 300, 400, SEAM)]
    assert walk == sorted(walk)
    assert walk[-1] > walk[0]


def test_welding_the_bottom_of_a_seam_moves_the_top_of_it():
    """The point of the whole module.

    The top of the seam has not been welded when the arc is a third of the way
    up, and it has still moved, because both sides are rotating about their
    folds. A cell that measured that point cold is now aiming at where it used
    to be.
    """
    hot = ThermalDistortion(seam_length_mm=SEAM)
    top_before_it_is_welded = hot.seam_shift_mm(SEAM, SEAM / 3.0)
    assert top_before_it_is_welded > 0.2


def test_the_seam_moves_further_the_higher_up_it_is():
    """The base holds the bottom of the corner. The top is free to swing."""
    hot = ThermalDistortion(seam_length_mm=SEAM)
    low = hot.seam_shift_mm(SEAM * 0.25, SEAM)
    high = hot.seam_shift_mm(SEAM, SEAM)
    assert high > low * 3.0


def test_shrinkage_lags_the_torch():
    """Weld metal does not finish pulling the instant it solidifies."""
    hot = ThermalDistortion(seam_length_mm=SEAM)
    just_behind = hot.cooled_length_mm(COOLING_LENGTH_MM)
    assert just_behind < COOLING_LENGTH_MM
    assert hot.cooled_length_mm(SEAM) > SEAM - COOLING_LENGTH_MM


# ------------------------------------------------------------------- the gap


def test_the_joint_pulls_shut_behind_the_arc():
    hot = ThermalDistortion(seam_length_mm=SEAM)
    assert hot.gap_change_mm(100.0, 100.0 + 2 * COOLING_LENGTH_MM) < -0.5


def test_the_joint_opens_slightly_just_ahead_of_the_arc():
    """Warm metal has expanded. Small, but it is why a read ahead is not the
    gap the torch will meet."""
    hot = ThermalDistortion(seam_length_mm=SEAM)
    assert hot.gap_change_mm(210.0, 200.0) > 0.0


def test_the_gap_is_undisturbed_far_ahead_of_the_arc():
    hot = ThermalDistortion(seam_length_mm=SEAM)
    assert hot.gap_change_mm(400.0, 20.0) == pytest.approx(0.0, abs=0.01)


# --------------------------------------------------- what look-ahead costs


def test_looking_ahead_once_is_not_enough():
    """The number that decides the architecture.

    The sensor reads the joint some distance in front of the torch. By the time
    the torch arrives, more seam has cooled and the sides have rotated further.
    At the lead this cell uses, that error is as large as the entire tracking
    accuracy, so the cell has to keep looking rather than measure once.
    """
    hot = ThermalDistortion(seam_length_mm=SEAM)
    assert hot.worst_drift_ahead_of_arc_mm(170.0) > 0.5


def test_a_shorter_lead_costs_less():
    hot = ThermalDistortion(seam_length_mm=SEAM)
    near = hot.worst_drift_ahead_of_arc_mm(20.0)
    far = hot.worst_drift_ahead_of_arc_mm(170.0)
    assert far > near * 3.0


def test_looking_at_the_arc_itself_costs_nothing_but_is_too_late():
    hot = ThermalDistortion(seam_length_mm=SEAM)
    assert hot.worst_drift_ahead_of_arc_mm(0.0) == pytest.approx(0.0, abs=1e-9)


def test_a_colder_weld_needs_less_chasing():
    """Turning the heat down is the other way to fight distortion."""
    cool = ThermalDistortion(weld=WeldPass(travel_mm_s=16.0), seam_length_mm=SEAM)
    hot = ThermalDistortion(weld=WeldPass(travel_mm_s=6.0), seam_length_mm=SEAM)
    assert hot.worst_drift_ahead_of_arc_mm(170.0) > cool.worst_drift_ahead_of_arc_mm(170.0)
