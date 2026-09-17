"""Tests for the forming error the press leaves on a folded side.

The numbers here are the reason the cell exists, so they are checked against what
the physics says rather than against whatever the code happens to return: a lean
is zero at the fold and worst at the rim, a bow is zero at both and worst in the
middle, and a seam made of two independently formed sides does not come out
straight.
"""

from __future__ import annotations

import math
import random

import pytest

from cell.forming import (
    CornerForming,
    PanelForming,
    plan_corner,
    plan_panel,
    plan_tray,
    plan_trays,
)

HEIGHT = 0.45


# ------------------------------------------------------------------ springback


def test_a_perfect_side_has_no_error_anywhere():
    panel = PanelForming()
    assert panel.worst_mm(HEIGHT) == pytest.approx(0.0)


def test_the_fold_holds_the_sheet_so_the_error_starts_at_zero():
    panel = PanelForming(lean_deg=0.6, bow_mm=2.0)
    assert panel.offset_m(0.0, HEIGHT) == pytest.approx(0.0, abs=1e-9)


def test_a_lean_is_worst_at_the_rim_and_grows_with_height():
    panel = PanelForming(lean_deg=0.5)
    quarter = panel.offset_m(HEIGHT * 0.25, HEIGHT)
    rim = panel.offset_m(HEIGHT, HEIGHT)
    assert rim > quarter > 0.0
    assert rim == pytest.approx(HEIGHT * math.tan(math.radians(0.5)))


def test_half_a_degree_of_springback_is_millimetres_at_the_rim():
    """The whole problem in one number: a tiny angle over a tall side."""
    panel = PanelForming(lean_deg=0.5)
    assert panel.worst_mm(HEIGHT) == pytest.approx(3.93, abs=0.05)


def test_the_lean_reverses_with_its_sign():
    out = PanelForming(lean_deg=0.4).offset_m(HEIGHT, HEIGHT)
    back = PanelForming(lean_deg=-0.4).offset_m(HEIGHT, HEIGHT)
    assert out == pytest.approx(-back)


# ----------------------------------------------------------------------- camber


def test_a_bow_is_worst_in_the_middle_and_gone_at_both_ends():
    panel = PanelForming(bow_mm=3.0)
    assert panel.offset_m(0.0, HEIGHT) == pytest.approx(0.0, abs=1e-9)
    assert panel.offset_m(HEIGHT, HEIGHT) == pytest.approx(0.0, abs=1e-9)
    assert panel.offset_m(HEIGHT / 2, HEIGHT) == pytest.approx(0.003, abs=1e-6)


def test_lean_and_bow_add_rather_than_replace_each_other():
    both = PanelForming(lean_deg=0.4, bow_mm=2.0).offset_m(HEIGHT / 2, HEIGHT)
    lean = PanelForming(lean_deg=0.4).offset_m(HEIGHT / 2, HEIGHT)
    bow = PanelForming(bow_mm=2.0).offset_m(HEIGHT / 2, HEIGHT)
    assert both == pytest.approx(lean + bow)


def test_the_offset_does_not_run_off_past_the_rim():
    panel = PanelForming(lean_deg=0.5)
    assert panel.offset_m(HEIGHT * 3, HEIGHT) == pytest.approx(
        panel.offset_m(HEIGHT, HEIGHT)
    )


def test_a_side_of_no_height_is_refused():
    with pytest.raises(ValueError):
        PanelForming(lean_deg=0.5).offset_m(0.1, 0.0)


# ------------------------------------------------- what this does to the seam


def test_the_two_sides_of_a_corner_move_the_seam_in_different_directions():
    corner = CornerForming(a=PanelForming(lean_deg=0.5), b=PanelForming(lean_deg=0.3))
    dx, dy = corner.seam_offset_m(HEIGHT, HEIGHT)
    assert dx > 0.0
    assert dy < 0.0


def test_pure_springback_leaves_the_seam_straight_only_tilted():
    """A lean alone tilts the seam. Touching one point would still correct it."""
    corner = CornerForming(a=PanelForming(lean_deg=0.7), b=PanelForming(lean_deg=-0.4))
    assert corner.seam_wander_mm(HEIGHT) < 0.02


def test_camber_makes_the_seam_wander_off_its_own_straight_line():
    """A bow cannot be corrected by touching the ends. It has to be followed."""
    corner = CornerForming(a=PanelForming(bow_mm=3.0), b=PanelForming())
    assert corner.seam_wander_mm(HEIGHT) == pytest.approx(3.0, abs=0.05)


def test_a_perfectly_formed_corner_has_a_seam_that_does_not_wander():
    assert CornerForming().seam_wander_mm(HEIGHT) == pytest.approx(0.0)


def test_bow_on_both_sides_wanders_by_their_diagonal():
    corner = CornerForming(a=PanelForming(bow_mm=3.0), b=PanelForming(bow_mm=4.0))
    assert corner.seam_wander_mm(HEIGHT) == pytest.approx(5.0, abs=0.05)


# ------------------------------------------------------------------- planning


def test_a_campaign_replays_identically():
    assert plan_trays(6, seed=4) == plan_trays(6, seed=4)


def test_different_seeds_give_different_parts():
    assert plan_trays(6, seed=4) != plan_trays(6, seed=5)


def test_every_side_of_a_tray_is_formed_independently():
    tray = plan_tray(random.Random(2))
    sides = {tray.right, tray.left, tray.front, tray.back}
    assert len(sides) == 4


def test_the_welded_corner_is_the_right_side_meeting_the_front():
    tray = plan_tray(random.Random(7))
    corner = tray.welded_corner()
    assert corner.a == tray.right
    assert corner.b == tray.front


def test_errors_stay_inside_what_a_press_actually_leaves():
    """A press this far out would be repaired, not worked around."""
    for tray in plan_trays(400, seed=1):
        for side in (tray.right, tray.left, tray.front, tray.back):
            assert abs(side.lean_deg) <= 0.30
            assert abs(side.bow_mm) <= 1.10


def test_turning_the_scale_up_makes_worse_parts():
    mild = plan_trays(200, seed=8, scale=1.0)
    rough = plan_trays(200, seed=8, scale=2.5)

    def worst(trays):
        return sum(t.welded_corner().seam_wander_mm(HEIGHT) for t in trays) / len(trays)

    assert worst(rough) > worst(mild) * 1.5


def test_a_negative_scale_is_refused():
    with pytest.raises(ValueError):
        plan_panel(random.Random(0), scale=-1.0)


def test_the_press_is_in_spec_and_the_weld_still_misses():
    """The reason this is a robot problem rather than a press problem.

    Nobody would rebuild a press to take a millimetre off the rim of a 450 mm
    flange. But a weld has to land within a few tenths of the seam, so what the
    press comfortably holds is still several times the tolerance that matters.
    """
    wanders = sorted(
        plan_corner(random.Random(s)).seam_wander_mm(HEIGHT) for s in range(300)
    )
    median = wanders[len(wanders) // 2]
    assert median < 1.0, "a press leaving more than this would be repaired instead"
    assert median > 0.3, "and it still strays further than a weld can tolerate"
    assert wanders[-1] < 2.0
