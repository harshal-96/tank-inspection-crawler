"""Tests for the supervisor's tools and output handling.

No network. The model is not exercised here; these cover the layer between it and
the database, which is where a wrong answer would become a wrong action.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session

from brain import tools
from brain.supervisor import plain_text
from service.db import engine
from service.models import Decision, DefectType, Verdict


@pytest.fixture()
def session(client):
    """Reuse the client fixture so the scratch database exists and is empty."""
    with Session(engine) as s:
        yield s


def _seed(client, n=3, with_truth=True):
    for i in range(n):
        serial = client.post("/api/parts", json={}).json()["serial"]
        client.post(
            "/api/measurements",
            json={
                "part_serial": serial,
                "detected_x": 0.0,
                "detected_y": (i + 1) / 1000.0,
                "confidence": 0.9,
                **({"true_x": 0.0, "true_y": 0.0} if with_truth else {}),
            },
        )
    return serial


# ------------------------------------------------------------------ output hygiene


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.99 mm — inside spec", "0.99 mm, inside spec"),
        ("drift 2.2 mm—locator wear", "drift 2.2 mm, locator wear"),
        ("range 1–3 mm", "range 1, 3 mm"),
        ("part P‑000042", "part P-000042"),
        ("it’s “fine”…", 'it\'s "fine"...'),
    ],
)
def test_plain_text_strips_typographic_punctuation(raw, expected):
    assert plain_text(raw) == expected


def test_plain_text_preserves_punctuation_after_space():
    # Regression: the collapse rule once deleted the punctuation it was tidying.
    assert plain_text("done .") == "done."
    assert plain_text("a , b") == "a, b"


def test_plain_text_leaves_plain_ascii_untouched():
    text = "Mean error 1.54 mm, p95 3.31 mm. Three parts over 3 mm."
    assert plain_text(text) == text


def test_no_em_dash_survives():
    assert "—" not in plain_text("a — b — c")


# -------------------------------------------------------------------------- tools


def test_dispatch_rejects_unknown_tool(session):
    assert "error" in tools.dispatch("not_a_tool", {}, session)


def test_dispatch_reports_bad_arguments_as_data(session):
    # A malformed call must come back as a tool result, not raise into the loop.
    result = tools.dispatch("get_part_detail", {"wrong_kwarg": 1}, session)
    assert "error" in result


def test_get_cell_status_shape_on_empty_cell(session):
    status = tools.get_cell_status(session)
    assert status["parts_total"] == 0
    assert status["correction_error_mm"]["n"] == 0
    assert status["drift"]["trending"] is False


def test_get_cell_status_counts_over_threshold(client, session):
    for dy_mm in (0.5, 2.0, 4.0, 5.0):
        serial = client.post("/api/parts", json={}).json()["serial"]
        client.post(
            "/api/measurements",
            json={
                "part_serial": serial,
                "detected_x": 0.0,
                "detected_y": dy_mm / 1000.0,
                "confidence": 0.9,
                "true_x": 0.0,
                "true_y": 0.0,
            },
        )

    correction = tools.get_cell_status(session)["correction_error_mm"]
    assert correction["n"] == 4
    assert correction["parts_over_3mm"] == 2  # 4.0 and 5.0
    assert correction["sub_mm_rate"] == 0.25  # only 0.5


def test_get_part_detail_unknown_serial(session):
    assert "error" in tools.get_part_detail(session, "P-nope")


def test_get_drift_reports_insufficient_data(client, session):
    _seed(client, n=3)
    drift = tools.get_drift(session)
    assert drift["available"] is False
    assert "Needs at least" in drift["reason"]


def test_list_defects_returns_only_defects(client, session):
    for verdict, defect in (("pass", "none"), ("defect", "porosity"), ("pass", "none")):
        serial = client.post("/api/parts", json={}).json()["serial"]
        client.post(
            "/api/inspections",
            json={
                "part_serial": serial,
                "image_path": "x.png",
                "verdict": verdict,
                "defect_type": defect,
            },
        )

    defects = tools.list_defects(session)["defects"]
    assert len(defects) == 1
    assert defects[0]["defect_type"] == DefectType.POROSITY.value


# ----------------------------------------------------------------------- decisions


def test_record_decision_persists_awaiting_acknowledgement(client, session):
    serial = _seed(client, n=1)
    result = tools.record_decision(
        session,
        action="recalibrate",
        rationale="Drift 2.17 mm and trending.",
        part_serial=serial,
        model="test-model",
    )
    assert result["recorded"] is True
    assert result["awaiting_acknowledgement"] is True

    stored = session.get(Decision, result["decision_id"])
    assert stored.action.value == "recalibrate"
    assert stored.acknowledged is False  # nothing acts without a human
    assert stored.model == "test-model"


def test_record_decision_rejects_unknown_action(session):
    assert "error" in tools.record_decision(session, action="explode", rationale="no")


def test_record_decision_rejects_unknown_part(session):
    result = tools.record_decision(
        session, action="rework", rationale="x", part_serial="P-nope"
    )
    assert "error" in result


def test_inspection_verdict_enum_matches_tool_filter(client, session):
    """list_defects filters on the enum, not a bare string."""
    serial = client.post("/api/parts", json={}).json()["serial"]
    client.post(
        "/api/inspections",
        json={
            "part_serial": serial,
            "image_path": "x.png",
            "verdict": Verdict.DEFECT.value,
            "defect_type": DefectType.UNDERCUT.value,
        },
    )
    assert len(tools.list_defects(session)["defects"]) == 1
