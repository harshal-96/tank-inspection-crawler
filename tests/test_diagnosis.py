"""The once-a-shift diagnosis: rows from stored seams, the study's question, the rule beside it."""

from __future__ import annotations

import json
from types import SimpleNamespace

from brain import diagnosis, nebius, tools
from cell import geometry as geo
from cell.drift import rule


def _track(bottom_off: float, top_off: float) -> dict:
    y0 = geo.nominal_joint()[1] * 1000.0
    return {"path": [{"along_mm": 10.0, "y_mm": y0 + bottom_off},
                     {"along_mm": 200.0, "y_mm": y0 + (bottom_off + top_off) / 2},
                     {"along_mm": 390.0, "y_mm": y0 + top_off}]}


def _response(content: str):
    message = SimpleNamespace(content=content, reasoning=None, reasoning_content=None,
                              tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))


class FakeAPI:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _response(self.reply)


def test_rows_are_offsets_from_the_nominal_joint_bottom_and_top():
    rows = diagnosis.rows_from_tracks([_track(0.1, 0.5), _track(-0.2, 0.3)])
    assert [r["part"] for r in rows] == [0, 1]
    assert abs(rows[0]["seam_bottom_mm"] - 0.1) < 1e-9
    assert abs(rows[0]["seam_top_mm"] - 0.5) < 1e-9
    assert abs(rows[0]["tilt_mm"] - 0.4) < 1e-9
    assert diagnosis.rows_from_tracks([{"path": []}]) == []


def test_a_walking_top_is_the_press_by_arithmetic_and_the_model_is_asked_the_same():
    # The top of the seam walks 0.15 mm a part while the bottom stays put.
    tracks = [_track(0.01 * ((-1) ** i), 0.15 * i) for i in range(20)]
    rows = diagnosis.rows_from_tracks(tracks)
    assert rule(rows) == "press"

    api = FakeAPI(json.dumps({"cause": "press", "confidence": 0.9, "evidence": "tilt trends"}))
    got = diagnosis.diagnose(rows, model="m", api=api)
    assert got["available"] and got["cause"] == "press" and got["rule_cause"] == "press"
    assert got["agree"] and got["parts"] == 20
    system = api.calls[0]["messages"][0]["content"]
    user = api.calls[0]["messages"][1]["content"]
    assert "times that scatter" in system          # the boundary sentence
    assert "Fitted over the whole run:" in user    # the fitted numbers
    assert "reference" not in user                 # no reference channel in the cell's data


def test_too_few_parts_is_said_rather_than_guessed():
    got = diagnosis.diagnose(diagnosis.rows_from_tracks([_track(0, 0)] * 3), model="m",
                             api=FakeAPI("{}"))
    assert not got["available"] and "8" in got["reason"]


def test_the_supervisor_can_call_it():
    assert "diagnose_drift" in tools.REGISTRY
    names = [t["function"]["name"] for t in tools.TOOL_SCHEMAS]
    assert "diagnose_drift" in names


def test_default_diagnosis_model_is_the_one_the_study_picked(monkeypatch):
    monkeypatch.delenv("CELL_DIAGNOSIS_MODEL", raising=False)
    assert nebius.diagnosis_model() == "deepseek-ai/DeepSeek-V4.1-Flash"
    monkeypatch.setenv("CELL_DIAGNOSIS_MODEL", "x/y")
    assert nebius.diagnosis_model() == "x/y"
