"""Tests for the tank half, which had none.

The suite was 292 tests and every one of them covered the welding cell, while
the tank is what the project now leads with. Worse, almost every bug found in
the tank was in the scoring and not in the model, and scoring is exactly the
kind of pure arithmetic a test pins down for free.

So each test here is a mistake that was actually made, written down so it
cannot come back. None of them need Blender, a network or a key.
"""

from __future__ import annotations

import pytest

from tank import adhesion
from tank.inspect_seam import SEAM_DEFECTS, SeamVerdict, _coerce, worst
from tank.probe_planner import (DEFAULT_BUDGET, Memory, Reading, WATCH_FRACTION,
                                _coerce as coerce_offsets, plan_with_floor,
                                sweep_rule)
from tank.score_survey import CAMERA_WORD, TRACE, score, truth_for

OFFSETS = [-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45]


# --------------------------------------------------------------- the verdict


def test_worst_ranks_a_crack_above_corrosion():
    assert worst(["corrosion", "crack", "porosity"]) == "crack"
    assert worst([]) == "none"
    assert worst(["not_a_defect"]) == "none"


def test_a_defect_naming_nothing_is_uncertain_not_a_pass():
    """It has not judged the frame. Turning that into a pass invents a verdict."""
    got = _coerce({"verdict": "defect", "defect_type": "none"})
    assert got.verdict == "uncertain"


def test_a_pass_cannot_carry_a_defect():
    got = _coerce({"verdict": "pass", "worst": "crack", "faults": ["crack"]})
    assert (got.verdict, got.defect, got.faults) == ("pass", "none", [])


def test_faults_outside_the_vocabulary_are_dropped_not_guessed():
    got = _coerce({"verdict": "defect", "faults": ["crack", "banana", "crack"],
                   "worst": "crack"})
    assert got.faults == ["crack"]


def test_the_fault_list_comes_back_ranked():
    got = _coerce({"verdict": "defect", "faults": ["corrosion", "crack"],
                   "worst": "crack"})
    assert got.faults == ["crack", "corrosion"]


def test_a_headline_absent_from_its_own_list_is_corrected_to_the_list():
    got = _coerce({"verdict": "defect", "faults": ["undercut"], "worst": "crack"})
    assert got.defect == "undercut"


def test_an_unreadable_reply_never_becomes_a_verdict():
    v = SeamVerdict("uncertain", "none", parse_error="boom")
    assert v.as_row()["verdict"] == "uncertain"
    assert v.as_row()["called"] == "none"


# ------------------------------------------------------------- the answer key


def _frame(*faults):
    return {"station": 0, "faults_in_view": [
        {"kind": k, "visible_to_a_camera": vis, "coverage": cov}
        for k, vis, cov in faults]}


def test_wall_loss_a_camera_can_see_is_called_corrosion():
    """The scene names a fault for its cause, the inspector for its look.

    Scored word against word rather than thing against thing, every correct
    call of rust read as an invention.
    """
    assert CAMERA_WORD["thinning"] == "corrosion"
    truth, seen, _ = truth_for(_frame(("thinning", True, 1.0)))
    assert truth == "corrosion" and seen == ["corrosion"]


def test_wall_loss_with_nothing_on_the_surface_is_not_charged_to_the_camera():
    truth, seen, grazed = truth_for(_frame(("thinning", False, 1.0)))
    assert (truth, seen, grazed) == ("none", [], [])


def test_a_sliver_in_the_corner_is_neither_a_defect_nor_an_invention():
    truth, seen, grazed = truth_for(_frame(("corrosion", True, TRACE / 2)))
    assert truth == "none" and seen == [] and grazed == ["corrosion"]


def test_naming_something_the_frame_holds_a_sliver_of_is_not_a_false_alarm():
    rows = [_row(truth="none", called="corrosion", faults=["corrosion"],
                 in_view=[], grazed=["corrosion"])]
    got = score(rows)
    assert got["false_alarms"] == 0
    assert got["called_a_sliver"] == 1


# ------------------------------------------------------------- the scoring


def _row(truth, called, faults, in_view, grazed=(), verdict="defect"):
    return {"verdict": verdict, "truth": truth, "called": called,
            "faults": list(faults), "in_view": list(in_view),
            "grazed": list(grazed), "latency_ms": 10, "retried_with_room": False}


def test_a_miss_a_misname_and_a_decline_are_three_different_things():
    rows = [
        _row("crack", "none", [], ["crack"]),                       # missed
        _row("crack", "porosity", ["porosity"], ["crack"]),         # misnamed
        _row("crack", "none", [], ["crack"], verdict="uncertain"),  # declined
    ]
    got = score(rows)
    assert got["missed"] == 1
    assert got["misnamed"] == 1
    assert got["declined"] == 1
    assert got["judged"] == 2


def test_finding_a_lesser_fault_that_is_really_there_is_counted_apart():
    rows = [_row("undercut", "corrosion", ["corrosion"], ["undercut", "corrosion"])]
    got = score(rows)
    assert got["named_a_lesser_fault_that_was_there"] == 1
    assert got["faults_found"] == 1
    assert got["invented"] == 0


def test_a_fault_in_neither_list_is_an_invention():
    rows = [_row("none", "crack", ["crack"], [], verdict="defect")]
    assert score(rows)["invented"] == 1


# --------------------------------------------------------- the probe planner


def test_the_rule_goes_back_to_a_reading_that_came_back_thin():
    memory = Memory([Reading(10, 0.30, 11.0, 11.0),
                     Reading(11, -0.45, 11.0 * WATCH_FRACTION - 0.5, 11.0)])
    assert sweep_rule(OFFSETS, 12, memory, 1) == [-0.45]


def test_the_rule_otherwise_spends_on_what_it_has_never_measured():
    memory = Memory([Reading(9, o, 11.0, 11.0) for o in OFFSETS if o != 0.45])
    assert sweep_rule(OFFSETS, 10, memory, 1) == [0.45]


def test_the_rule_is_deterministic():
    memory = Memory([Reading(1, -0.15, 11.0, 11.0)])
    first = sweep_rule(OFFSETS, 2, memory, 2)
    assert first == sweep_rule(OFFSETS, 2, memory, 2)


def test_offsets_the_arm_cannot_reach_are_dropped_not_snapped_from_far_away():
    assert coerce_offsets({"offsets": [0.15, 9.9]}, OFFSETS, 2) == [0.15]
    assert coerce_offsets({"offsets": ["-0.30"]}, OFFSETS, 2) == [-0.30]
    assert coerce_offsets({"offsets": [0.151]}, OFFSETS, 2) == [0.15]


def test_the_floor_hands_a_single_probe_to_the_rule_without_calling_anything():
    """With one measurement there is nothing for a model to add, and asking
    anyway would spend a call and a second of arm time to be told the same."""
    memory = Memory([Reading(1, -0.15, 11.0, 11.0)])
    plan = plan_with_floor(2, OFFSETS, 11.0, 0.6, "sound seam", memory, budget=1)
    assert plan.offsets == sweep_rule(OFFSETS, 2, memory, 1)
    assert plan.model is None


def test_a_reading_is_a_fraction_of_nominal_not_a_depth():
    assert Reading(1, 0.0, 5.5, 11.0).fraction == pytest.approx(0.5)


def test_the_default_budget_is_smaller_than_the_arm_can_reach():
    assert 1 <= DEFAULT_BUDGET < len(OFFSETS)


# ------------------------------------------------------------- the adhesion


def test_the_wheelbase_is_read_from_the_builder_not_copied():
    """Two copies of a wheelbase means one of them going stale."""
    got = adhesion.scene_constants("WHEEL_AT", "COURSE_MM", "BEAD_R")
    assert len(got["WHEEL_AT"]) == 4
    assert got["COURSE_MM"][0] > got["COURSE_MM"][-1]   # courses thin with height
    assert 0.0 < got["BEAD_R"] < 0.05


def test_asking_for_a_constant_that_is_not_there_fails_loudly():
    with pytest.raises(KeyError):
        adhesion.scene_constants("NOT_A_REAL_CONSTANT")


def test_a_thin_plate_cannot_carry_the_rated_pull():
    assert adhesion.plate_derate(12.0) == 1.0
    assert adhesion.plate_derate(6.0) < 1.0


def test_a_coating_costs_grip_without_being_catastrophic():
    """Modelled as inverse square from contact, half a millimetre of paint
    appears to remove ninety percent of the grip. Magnets do not do that."""
    assert adhesion.gap_derate(0.2) == pytest.approx(1.0)
    half = adhesion.gap_derate(0.6)
    assert 0.5 < half < 0.9
    assert adhesion.gap_derate(1.2) < half


def test_it_holds_on_clean_thick_plate_and_lets_go_under_corrosion():
    clean = adhesion.budget(6.8, 0.35, 3.0, 0.2, 12.0)
    fouled = adhesion.budget(6.8, 0.35, 3.0, 1.2, 6.0)
    assert clean["margin"] > 1.0
    assert fouled["margin"] < 1.0


def test_the_tether_is_counted_because_it_is_not_nothing():
    assert adhesion.budget(6.8, 0.35, 3.0, 0.2, 12.0)["tether_n"] > 0
