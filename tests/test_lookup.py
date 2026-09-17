"""Tests for looking something up outside the cell.

No network here. What is checked is that a missing key or a failed request comes
back as data the supervisor can talk about, rather than as an exception that takes
the cell down, and that a result always arrives with its sources attached.
"""

from __future__ import annotations

import brain.lookup as lookup


def test_a_missing_key_is_reported_not_raised(monkeypatch):
    """A supervisor that falls over because a search key is absent is worse than
    one that says it could not check."""
    monkeypatch.setattr(lookup, "api_key",
                        lambda: (_ for _ in ()).throw(lookup.MissingTavilyKey("no key")))
    result = lookup.search("anything")
    assert result.error
    assert result.answer is None


def test_a_failed_request_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(lookup, "api_key", lambda: "test-key")

    def explode(*args, **kwargs):
        raise TimeoutError("the internet went away")

    monkeypatch.setattr(lookup.httpx, "post", explode)
    result = lookup.search("anything")
    assert "TimeoutError" in (result.error or "")


def test_results_carry_their_sources(monkeypatch):
    monkeypatch.setattr(lookup, "api_key", lambda: "test-key")

    class Reply:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "answer": "Up to about 2 mm.",
                "results": [
                    {"title": "A paper", "url": "https://example.org/a", "content": "text"},
                ],
            }

    monkeypatch.setattr(lookup.httpx, "post", lambda *a, **k: Reply())
    result = lookup.search("root gap limit")
    assert result.answer == "Up to about 2 mm."
    assert result.sources[0].url == "https://example.org/a"
    assert "example.org" in result.sources[0].cite()


def test_the_tool_result_tells_the_model_not_to_pass_it_off_as_a_measurement(monkeypatch):
    """The cell measures. A search result is evidence about the world, and the
    difference matters when the answer decides whether a part ships."""
    monkeypatch.setattr(lookup, "api_key", lambda: "test-key")

    class Reply:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"answer": "Something.", "results": []}

    monkeypatch.setattr(lookup.httpx, "post", lambda *a, **k: Reply())
    payload = lookup.search("anything").as_tool_result()
    assert "measurement" in payload["note"]
    assert "Cite" in payload["note"]


def test_an_errored_lookup_does_not_pretend_to_have_an_answer(monkeypatch):
    monkeypatch.setattr(lookup, "api_key",
                        lambda: (_ for _ in ()).throw(lookup.MissingTavilyKey("no key")))
    payload = lookup.search("anything").as_tool_result()
    assert "error" in payload
    assert "answer" not in payload


def test_the_query_is_not_padded_with_vague_words(monkeypatch):
    """Padding a technical question with general words turned it into a search for
    local welding businesses. The query goes as written."""
    monkeypatch.setattr(lookup, "api_key", lambda: "test-key")
    seen = {}

    class Reply:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"answer": "", "results": []}

    def capture(url, json=None, timeout=None):
        seen.update(json or {})
        return Reply()

    monkeypatch.setattr(lookup.httpx, "post", capture)
    lookup.search("root gap limit on 2 mm sheet")
    assert seen["query"] == "root gap limit on 2 mm sheet"


def test_the_supervisor_can_reach_it_as_a_tool():
    from brain.tools import REGISTRY, TOOL_SCHEMAS

    assert "look_up_standard" in REGISTRY
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert "look_up_standard" in names
