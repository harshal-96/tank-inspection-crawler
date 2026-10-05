"""The per-part look-again decision, scored for the arithmetic rule as well as the model.

The smallest Nemotron's 88 of 89 only means something beside what the cell's
own limits get on the same points. These run the rule arm of
scripts/eval_router_truth.py offline and pin its counts. No test here may call
a model, so the transport is replaced with one that fails the test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from brain import nebius
from scripts import eval_router_truth as ert

ROOT = Path(__file__).resolve().parent.parent
ANSWERS = ROOT / "data" / "router_truth_eval.jsonl"
CAMPAIGN = ROOT / "data" / "campaign50.jsonl"
RULE_RECORD = ROOT / "data" / "router_rule_eval.json"
RERENDER = ROOT / "data" / "cap50v3" / "manifest.json"


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the rule arm called a model")
    monkeypatch.setattr(nebius, "chat", refuse)
    monkeypatch.setattr(nebius, "client", refuse)


def _nano_rows() -> list[dict]:
    rows = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [r for r in rows if r["arm"] == "nano"]


def test_the_88_of_89_is_39_replayed_and_50_first_view_points():
    nano = _nano_rows()
    assert len(nano) == 89
    assert sum(1 for r in nano if r["said_look_again"] == r["should_look_again"]) == 88
    assert sum(1 for r in nano if r["kind"] == "replayed") == 39
    first = [r for r in nano if r["kind"] == "first-view"]
    assert len(first) == 50 and sum(r["should_look_again"] for r in first) == 25


def test_rule_ties_nano_on_the_replayed_points():
    nano = [r for r in _nano_rows() if r["kind"] == "replayed"]
    points = list(ert.decision_points(CAMPAIGN))
    assert len(points) == len(nano) == 39
    rule_right = nano_right = 0
    for point, row in zip(points, nano, strict=True):
        assert point["trial"] == row["trial"]
        assert point["gain_mm"] == pytest.approx(row["gain_mm"], abs=1e-9)
        rule = ert.rule_says_look_again(point["evidence"], point["remaining"])
        rule_right += rule == point["should_look_again"]
        nano_right += row["said_look_again"] == row["should_look_again"]
    assert (rule_right, nano_right) == (39, 39)


def test_rule_gets_47_of_50_first_view_points_from_the_kept_signals():
    record = json.loads(RULE_RECORD.read_text(encoding="utf-8"))
    nano = [r for r in _nano_rows() if r["kind"] == "first-view"]
    kept = record["first_view_points"]
    assert [p["trial"] for p in kept] == [r["trial"] for r in nano]
    rule_right = only_rule = only_nano = 0
    for p, row in zip(kept, nano, strict=True):
        look = ({"view_deg": p["view_deg"], "measurement_failed": "refused"} if p["failed"]
                else {"view_deg": p["view_deg"], "signals": p["signals"]})
        rule = ert.rule_says_look_again([look], p["remaining"]) == row["should_look_again"]
        model = row["said_look_again"] == row["should_look_again"]
        rule_right += rule
        only_rule += rule and not model
        only_nano += model and not rule
    assert (rule_right, only_rule, only_nano) == (47, 1, 3)
    assert ert.mcnemar_exact(only_rule, only_nano) == pytest.approx(0.625)
    assert record["compare"]["all"]["rule_right"] == 86
    assert record["compare"]["all"]["nano_right"] == 88


def test_the_rule_accepts_a_good_view_in_the_shape_the_rule_reads():
    point = next(ert.decision_points(CAMPAIGN))
    assert not ert.rule_says_look_again(point["evidence"], point["remaining"])


def test_the_rule_accepts_a_good_view_as_the_cell_passes_it():
    """It read only flat signals, so on the cell's nested looks the fallback
    asked for another look on every part. Fixed in brain/viewplanner.py."""
    point = next(ert.decision_points(CAMPAIGN))
    accept, _ = ert.rule_decision(point["evidence"], point["remaining"])
    assert accept


def test_a_limit_the_reason_compares_against_is_not_a_reading():
    point = {"trial": "t", "remaining": [0.0],
             "evidence": [{"view_deg": 22.5, "signals": {
                 "included_angle_deg": 89.5, "inlier_balance": 0.508,
                 "face_a_fit_rms_mm": 0.1, "face_b_fit_rms_mm": 0.1, "path_length_mm": 360.0}}]}
    assert ert.quoted_readings(point, ["inlier_balance exceeds 0.60, "
                                       "included_angle_deg is within 5 of 90"]) == []
    seen = ert.quoted_readings(point, ["inlier_balance is 0.508", "inlier_balance (0.65)"])
    assert [(q["quoted"], q["moves"]) for q in seen] == [(0.508, False), (0.65, True)]


def test_quoted_readings_move_the_rule_on_no_point():
    record = json.loads(RULE_RECORD.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    reasons: dict[str, list[str]] = {}
    for row in rows:
        if row["kind"] == "first-view":
            reasons.setdefault(row["trial"], []).append(row["reason"] or "")
    quoted = []
    for p in record["first_view_points"]:
        if not p["failed"]:
            point = {"trial": p["trial"], "remaining": p["remaining"],
                     "evidence": [{"view_deg": p["view_deg"], "signals": p["signals"]}]}
            quoted += ert.quoted_readings(point, reasons[p["trial"]])
    assert len(quoted) == record["quoted_readings"]["checked_one_at_a_time"] == 363
    assert not any(q["moves"] for q in quoted)
    assert record["quoted_readings"]["points_where_the_rule_moves"] == []


def test_the_comparison_stops_when_the_answer_key_does_not_pair(tmp_path):
    record = json.loads(RULE_RECORD.read_text(encoding="utf-8"))
    flipped = record["first_view_points"][0]
    flipped["should_look_again_rebuilt"] = not flipped["should_look_again_rebuilt"]
    kept = tmp_path / "rule.json"
    kept.write_text(json.dumps(record), encoding="utf-8")
    args = argparse.Namespace(record=str(ANSWERS), campaign=str(CAMPAIGN),
                              manifest=str(tmp_path / "no_render.json"), rule_out=str(kept))
    with pytest.raises(SystemExit, match="do not pair"):
        ert.run_rule_arm(args)


def test_misses_are_listed_for_every_look():
    good = {"view_deg": 0.0, "included_angle_deg": 90.0, "inlier_balance": 0.8,
            "face_a_fit_rms_mm": 0.1, "face_b_fit_rms_mm": 0.1, "path_length_mm": 380.0}
    edge = {**good, "view_deg": 45.0, "inlier_balance": 0.15}
    assert ert.misses_text([edge]) == "balance 0.15"
    assert ert.misses_text([edge, good]) == "45 deg: balance 0.15; 0 deg: none"


def test_exact_mcnemar():
    assert ert.mcnemar_exact(0, 0) == 1.0
    assert ert.mcnemar_exact(1, 3) == pytest.approx(10 / 16)
    assert ert.mcnemar_exact(0, 5) == pytest.approx(2 / 32)


@pytest.mark.skipif(not RERENDER.exists(), reason="the re-render is not in the repository")
def test_kept_signals_match_the_re_render():
    record = json.loads(RULE_RECORD.read_text(encoding="utf-8"))
    rebuilt = {p["trial"]: p for p in ert.first_view_points(RERENDER)}
    for kept in record["first_view_points"]:
        point = rebuilt[kept["trial"]]
        look = point["evidence"][0]
        assert kept["failed"] == bool(look.get("measurement_failed"))
        if not kept["failed"]:
            assert kept["signals"] == {k: look["signals"][k] for k in ert.LIMIT_SIGNALS}
        assert kept["should_look_again_rebuilt"] == point["should_look_again"]
