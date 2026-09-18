"""The one call policy every model goes through, and what happens when a reply is unusable.

These run against a fake endpoint. What they pin down is behaviour that went
wrong silently before: an API option rejected by one model scored as that
model failing, a reasoning model's empty answer scored as a wrong verdict, and a
planner reply that would not parse turned into a decision to weld.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

from brain import advisor, inspector, nebius, viewplanner
from cell.limits import PROBLEM_MM, WELDABLE_MM


def _response(content, completion_tokens=50, reasoning=None):
    message = SimpleNamespace(content=content, reasoning=reasoning, reasoning_content=None,
                              tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=10,
                                                 completion_tokens=completion_tokens))


def _bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "http://test/v1/chat/completions")
    return BadRequestError(message, response=httpx.Response(400, request=request), body=None)


class FakeAPI:
    """Replays scripted replies and records every call's options."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


MESSAGES = [{"role": "user", "content": "hello"}]


def test_reasoning_effort_is_dropped_when_the_model_rejects_it():
    api = FakeAPI(_bad_request("reasoning_effort is not supported"), _response('{"ok": 1}'))
    reply = nebius.chat(MESSAGES, model="m", reasoning_effort="none", api=api)
    assert reply.raw == '{"ok": 1}'
    assert "reasoning_effort" in api.calls[0]
    assert "reasoning_effort" not in api.calls[1]
    assert reply.dropped_option == "reasoning_effort" and reply.effort == "default"


def test_other_bad_requests_still_raise():
    api = FakeAPI(_bad_request("image input not supported"))
    with pytest.raises(BadRequestError):
        nebius.chat(MESSAGES, model="m", reasoning_effort="none", api=api)


def test_a_reply_spent_thinking_is_asked_again_with_room():
    api = FakeAPI(_response("", completion_tokens=400, reasoning="let me think"),
                  _response('{"cause": "press"}', completion_tokens=30))
    reply = nebius.chat(MESSAGES, model="m", max_tokens=400, parses=nebius.extract_json, api=api)
    assert reply.retried_with_room
    assert api.calls[1]["max_tokens"] == nebius.THINKING_TOKENS
    assert json.loads(reply.raw)["cause"] == "press"


def test_a_reply_that_parses_is_not_retried():
    api = FakeAPI(_response('{"cause": "none"}', completion_tokens=400))
    reply = nebius.chat(MESSAGES, model="m", max_tokens=400, parses=nebius.extract_json, api=api)
    assert not reply.retried_with_room and len(api.calls) == 1


def test_a_short_wrong_reply_is_not_retried():
    # No reasoning field and budget not exhausted: the model answered badly,
    # which is a result, not an envelope problem.
    api = FakeAPI(_response("I cannot tell.", completion_tokens=8))
    reply = nebius.chat(MESSAGES, model="m", max_tokens=400, parses=nebius.extract_json, api=api)
    assert not reply.retried_with_room and len(api.calls) == 1


def test_extract_json_reads_fenced_and_prosed_replies():
    assert nebius.extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert nebius.extract_json('Sure: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        nebius.extract_json("no object here")


GOOD_VIEW = {"view_deg": 0.0, "included_angle_deg": 90.4, "inlier_balance": 0.8,
             "face_a_fit_rms_mm": 0.2, "face_b_fit_rms_mm": 0.3, "path_length_mm": 360.0,
             "joint_mm": [1.0, 2.0]}
EDGE_ON_VIEW = {"view_deg": 0.0, "included_angle_deg": 71.0, "inlier_balance": 0.2,
                "face_a_fit_rms_mm": 0.9, "face_b_fit_rms_mm": 0.3, "path_length_mm": 360.0,
                "joint_mm": [1.0, 2.0]}


def test_planner_prompt_carries_the_numbers_the_study_proved():
    assert "within 5 of 90" in viewplanner.THRESHOLD_TEXT
    assert "at least 0.60" in viewplanner.THRESHOLD_TEXT
    assert "under 0.5 mm" in viewplanner.THRESHOLD_TEXT
    assert "at least 300" in viewplanner.THRESHOLD_TEXT
    assert viewplanner.view_meets_rule(GOOD_VIEW)
    assert not viewplanner.view_meets_rule(EDGE_ON_VIEW)
    assert not viewplanner.view_meets_rule({})


def test_unreadable_planner_reply_falls_back_to_the_rule(monkeypatch):
    monkeypatch.setattr(nebius, "client", lambda: FakeAPI(_response("garbage", 5)))
    accepted = viewplanner.plan_next_view([GOOD_VIEW], [22.5, 45.0], model="m")
    assert accepted.accept and accepted.parse_error

    monkeypatch.setattr(nebius, "client", lambda: FakeAPI(_response("garbage", 5)))
    look = viewplanner.plan_next_view([EDGE_ON_VIEW], [22.5, 45.0], model="m")
    assert not look.accept and look.next_view_deg == 45.0 and look.parse_error


def test_planner_sends_the_threshold_unless_a_study_switches_it_off(monkeypatch):
    api = FakeAPI(_response('{"accept": true, "reason": "fine"}'))
    monkeypatch.setattr(nebius, "client", lambda: api)
    viewplanner.plan_next_view([GOOD_VIEW], [22.5], model="m")
    assert "within 5 of 90" in api.calls[0]["messages"][0]["content"]

    api = FakeAPI(_response('{"accept": true, "reason": "fine"}'))
    monkeypatch.setattr(nebius, "client", lambda: api)
    viewplanner.plan_next_view([GOOD_VIEW], [22.5], model="m", threshold=False)
    assert "within 5 of 90" not in api.calls[0]["messages"][0]["content"]


def test_inspector_framing_and_window_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("BEAD_SOURCE", raising=False)
    monkeypatch.delenv("BEAD_WINDOW", raising=False)
    assert inspector.default_framing() == "rendering"
    assert inspector.bead_window() == inspector.BEAD_WINDOW

    monkeypatch.setenv("BEAD_SOURCE", "photograph")
    monkeypatch.setenv("BEAD_WINDOW", "none")
    assert inspector.default_framing() == "photograph"
    assert inspector.bead_window() is None
    assert "rendering" not in inspector.FRAMINGS["photograph"]
    assert "uncertain" in inspector.FRAMINGS["photograph"]

    monkeypatch.setenv("BEAD_WINDOW", "10,20,300,400")
    assert inspector.bead_window() == (10, 20, 300, 400)
    monkeypatch.setenv("BEAD_WINDOW", "300,20,10,400")
    with pytest.raises(ValueError):
        inspector.bead_window()


def test_a_brief_without_a_traceable_citation_is_marked():
    brief = advisor.Brief(topic="gap", query="q", what_the_sources_say="Clause 5 says 3 mm.",
                          confidence="high", sources=[{"n": "1", "title": "t", "url": "u"}])
    advisor.check_citations(brief)
    assert not brief.citations_ok and brief.confidence == "low" and "citation" in brief.caveat

    cited = advisor.Brief(topic="gap", query="q", what_the_sources_say="Clause 5 says 3 mm [1].",
                          confidence="high", sources=[{"n": "1", "title": "t", "url": "u"}])
    advisor.check_citations(cited)
    assert cited.citations_ok and cited.confidence == "high"

    wrong = advisor.Brief(topic="gap", query="q", what_the_sources_say="See [4].",
                          sources=[{"n": "1", "title": "t", "url": "u"}])
    assert not advisor.check_citations(wrong).citations_ok


def test_the_supervisor_states_the_limits_the_cell_uses():
    from brain.supervisor import SYSTEM_PROMPT

    wanted = f"Under {WELDABLE_MM:g} mm is good. Over {PROBLEM_MM:g} mm is a problem."
    assert wanted in SYSTEM_PROMPT
