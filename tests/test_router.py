"""Tests for sending each decision to the right model.

The routing itself has to be deterministic, cheap and auditable, so it is tested
without touching the network. What the models then DO with the decisions is
measured separately by the evaluation scripts, because that needs ground truth
and cannot be asserted in a unit test.
"""

from __future__ import annotations

from brain.router import (
    NANO,
    SIZES,
    assess,
    tier_of,
)


def look(view_deg=0.0, x=4402.0, y=-602.0, balance=0.9, angle=90.0, failed=None):
    return {
        "view_deg": view_deg,
        "joint_mm": None if failed else [x, y],
        "failed": failed,
        "signals": {} if failed else {
            "inlier_balance": balance,
            "included_angle_deg": angle,
            "face_a_fit_rms_mm": 0.15,
            "face_b_fit_rms_mm": 0.16,
        },
    }


# -------------------------------------------------------------------- the sizes


def test_every_size_is_an_nvidia_model():
    assert all(model.startswith("nvidia/") for model in SIZES)


def test_every_size_has_a_name():
    assert {tier_of(m) for m in SIZES} == {"nano", "lightning", "super", "ultra"}


def test_an_unknown_model_is_not_silently_given_a_tier():
    assert tier_of("someone-elses/model") == "unknown"


# ------------------------------------------------- the default is the small one


def test_a_clean_single_view_goes_to_the_smallest_model():
    """Measured against ground truth, the smallest model is the right one here."""
    assert assess([look()]).model == NANO


def test_two_agreeing_views_still_go_to_the_smallest_model():
    """Agreement is not difficulty. Nothing about a second confirming view calls
    for a larger model."""
    route = assess([look(0.0), look(22.5, x=4402.4)])
    assert route.model == NANO


def test_a_marginal_but_consistent_view_does_not_escalate():
    """An earlier version escalated on low balance alone and scored worse for it.

    A single view with a thin second face is still a single view; the fusion has
    nothing to argue with, and a larger model asked about it just spends a robot
    move.
    """
    assert assess([look(balance=0.30, angle=78.0)]).model == NANO


def test_no_looks_yet_still_returns_a_usable_route():
    route = assess([])
    assert route.model in SIZES


# ------------------------------------------------------ nothing escalates any more


def test_a_refused_view_plus_disagreement_still_goes_to_the_smallest_model():
    """The 120B scores worst on this question, so sending hard evidence to it
    would cost robot moves. The route records the refusal instead."""
    route = assess([
        look(0.0),
        look(22.5, x=4460.0, y=-518.0),
        look(-45.0, failed="not the two faces of the weld"),
    ])
    assert route.model == NANO
    assert "1 refused" in route.describe()


def test_two_refused_views_still_go_to_the_smallest_model():
    route = assess([
        look(0.0, failed="one face only"),
        look(22.5, failed="not the two faces of the weld"),
    ])
    assert route.model == NANO
    assert "2 refused" in route.describe()


# ------------------------------------------------------------------ the record


def test_the_route_says_why_it_chose_what_it_chose():
    """An operator has to be able to ask why a decision went where it did."""
    route = assess([look()])
    assert route.reasons
    assert route.tier in route.describe()


def test_routing_costs_nothing_but_arithmetic():
    """No network, no model call. If routing needed a model it would have bought
    nothing."""
    for _ in range(500):
        assess([look(), look(22.5, x=4402.2)])
