"""The drift question: parsing and scoring a reply honestly."""

from cell import drift


class TestTheAnswerSpaceIsEnforced:
    """A reply outside the four causes is a third outcome, not a wrong diagnosis.

    Scoring by string comparison collapses three different failures into one. A
    model that cannot be parsed, a model that refuses, and a model that invents a
    category it was never offered are all "not the right word", and counting them
    together as wrong diagnoses both flatters the arms that fail loudly and hides
    the ones failing in an interesting way. On the boundary runs the 120B twice
    answered "reference", which is a column in the data rather than a cause.
    """

    def test_a_named_cause_survives_intact(self):
        got = drift.parse_verdict('{"cause": "press", "confidence": 0.8}')
        assert got["cause"] == "press"
        assert got["confidence"] == 0.8

    def test_case_and_padding_do_not_change_the_answer(self):
        assert drift.parse_verdict('{"cause": "  PRESS "}')["cause"] == "press"

    def test_prose_around_the_json_is_tolerated(self):
        raw = 'Looking at the tilt column, I think:\n{"cause": "fixture"}\nHope that helps.'
        assert drift.parse_verdict(raw)["cause"] == "fixture"

    def test_a_cause_nobody_offered_is_not_scored_as_a_wrong_diagnosis(self):
        got = drift.parse_verdict('{"cause": "reference", "confidence": 0.9}')
        assert got["cause"] == drift.CAUSE_OUT_OF_VOCABULARY
        assert got["said_instead"] == "reference"

    def test_the_allowed_set_narrows_with_the_question(self):
        """Camera is a real cause, but not when only three were offered."""
        assert drift.parse_verdict('{"cause": "camera"}')["cause"] == "camera"
        narrowed = drift.parse_verdict('{"cause": "camera"}', drift.SCENARIOS)
        assert narrowed["cause"] == drift.CAUSE_OUT_OF_VOCABULARY

    def test_unparseable_is_distinct_from_out_of_vocabulary(self):
        assert drift.parse_verdict("no json here at all")["cause"] == drift.CAUSE_UNPARSED
        assert drift.parse_verdict("")["cause"] == drift.CAUSE_UNPARSED
        assert drift.parse_verdict('{"cause": ')["cause"] == drift.CAUSE_UNPARSED

    def test_json_that_is_not_an_object_does_not_crash(self):
        assert drift.parse_verdict('["press"]')["cause"] == drift.CAUSE_UNPARSED

    def test_every_real_scenario_is_accepted(self):
        for scenario in drift.SCENARIOS_WITH_CAMERA:
            assert drift.parse_verdict(f'{{"cause": "{scenario}"}}')["cause"] == scenario
