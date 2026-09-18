"""The error budget, and the confound it exists to point at.

Two things here are easy to get wrong and expensive to get wrong quietly:
combining systematic terms as though they were random, which flatters the total,
and labelling a published figure as measured, which flatters the work. The first
version of the module did the second one in its own laser row.
"""

from __future__ import annotations

import math

import pytest

from cell.drift import build_stream, discriminators
from cell.error_budget import (
    INDUSTRY_REQUIREMENT_MM,
    Budget,
    Term,
    budgets,
    summary,
)


def test_systematic_terms_add_and_random_terms_go_in_quadrature():
    """The distinction is the whole point: one averages down, the other does not."""
    b = Budget("test", [
        Term("a", 3.0, "random", "measured here", True, ""),
        Term("b", 4.0, "random", "measured here", True, ""),
        Term("c", 1.0, "systematic", "published range", False, ""),
        Term("d", 2.0, "systematic", "published range", False, ""),
    ])
    assert b.random_mm() == pytest.approx(5.0)       # 3-4-5
    assert b.systematic_mm() == pytest.approx(3.0)   # 1 + 2
    assert b.total_mm() == pytest.approx(8.0)


def test_a_systematic_term_is_never_treated_as_averaging_down():
    systematic = Term("x", 1.0, "systematic", "published range", False, "")
    random = Term("y", 1.0, "random", "measured here", True, "")
    assert not systematic.averages_down
    assert random.averages_down


def test_the_quoted_figure_is_smaller_than_the_whole_chain():
    """The point of the study. If this ever fails the budget has lost its teeth."""
    rows = budgets()
    quoted = rows[0]
    assert len(quoted.terms) == 1, "the flattering row should hold exactly one term"
    for other in rows[1:]:
        assert other.total_mm() > quoted.total_mm()


def test_every_term_says_where_its_number_came_from():
    """A budget whose provenance is invisible is one nobody can argue with."""
    allowed = {"measured here", "published range", "specification"}
    for budget in budgets():
        for term in budget.terms:
            assert term.source in allowed
            assert term.note, f"{term.name} has no explanation"


def test_the_laser_row_does_not_claim_to_have_been_measured():
    """It is a published figure for a sensor this repository never ran."""
    laser = next(b for b in budgets() if b.label.startswith("Laser"))
    tracking = next(t for t in laser.terms if t.name == "Seam tracking")
    assert tracking.source == "published range"
    assert tracking.cite


def test_hand_eye_calibration_dominates_and_cannot_be_fixed_here():
    realistic = next(b for b in budgets() if b.label.startswith("Typical stereo"))
    hand_eye = next(t for t in realistic.terms if t.name == "Hand eye calibration")
    assert hand_eye.kind == "systematic"
    assert not hand_eye.fixable
    assert summary()["dominant_unmeasured"] == "Hand eye calibration"


def test_only_the_right_sensor_meets_the_requirement():
    rows = {b.label: b.total_mm() for b in budgets()}
    stereo = next(v for k, v in rows.items() if k.startswith("Typical stereo"))
    laser = next(v for k, v in rows.items() if k.startswith("Laser"))
    assert stereo > INDUSTRY_REQUIREMENT_MM
    assert laser < INDUSTRY_REQUIREMENT_MM


def test_compensating_the_thermal_term_is_worth_more_than_it_looks():
    """It is systematic, so it comes off the total one for one."""
    uncompensated = next(b for b in budgets() if b.label.startswith("Perfect sensor"))
    thermal = next(t for t in uncompensated.terms if t.name == "Thermal distortion")
    assert thermal.kind == "systematic"
    assert thermal.mm > 0.4


# ------------------------------------------------------------------- the confound


def test_a_drifting_camera_is_invisible_in_the_seam_alone():
    """The reason the reference feature exists.

    A fixture creeping moves the part; a camera creeping moves the measurement.
    Both show as the seam translating with the tilt flat, and the discriminators
    that separate press from fixture cannot separate these two at all.
    """
    for seed in (110, 248, 375, 520):
        fixture = discriminators(build_stream(30, "fixture", seed)[0])
        camera = discriminators(build_stream(30, "camera", seed)[0])
        assert fixture["tilt_ratio"] == pytest.approx(camera["tilt_ratio"], abs=0.01)
        # Both trend, and neither tilts.
        assert fixture["shift_ratio"] > 2.0 and camera["shift_ratio"] > 2.0
        assert fixture["tilt_ratio"] < 2.0 and camera["tilt_ratio"] < 2.0


def test_the_reference_separates_them():
    for seed in (110, 248, 375, 520):
        fixture = discriminators(build_stream(30, "fixture", seed, reference=True)[0])
        camera = discriminators(build_stream(30, "camera", seed, reference=True)[0])
        assert camera["reference_ratio"] > 2.0, "a moved camera should move the reference"
        assert fixture["reference_ratio"] < 2.0, "a moved fixture should not"


def test_nothing_but_the_camera_can_move_the_reference():
    """It is bolted to the machine base, so the press and the fixture cannot."""
    for scenario in ("press", "fixture", "none"):
        d = discriminators(build_stream(30, scenario, 4242, reference=True)[0])
        assert d["reference_ratio"] < 2.0, f"{scenario} moved the reference"


def test_the_reference_column_only_appears_when_asked_for():
    """Streams without it must be byte-identical to the ones already scored."""
    plain, _ = build_stream(30, "press", 110)
    assert "reference_mm" not in plain[0]

    from cell.drift import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_REFERENCE, prompt_for
    assert prompt_for(plain) is SYSTEM_PROMPT
    withref, _ = build_stream(30, "press", 110, reference=True)
    assert prompt_for(withref) is SYSTEM_PROMPT_WITH_REFERENCE


def test_an_unknown_scenario_is_refused():
    with pytest.raises(ValueError):
        build_stream(30, "gremlins", 1)


def test_quadrature_is_not_accidentally_a_plain_sum():
    """Guards the arithmetic itself, which is easy to 'simplify' wrongly."""
    b = Budget("t", [Term(f"t{i}", 1.0, "random", "measured here", True, "")
                     for i in range(4)])
    assert b.random_mm() == pytest.approx(2.0)
    assert b.random_mm() != pytest.approx(4.0)
    assert b.random_mm() == pytest.approx(math.sqrt(4))
