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


class TestTheBoundaryTestHasNoHoleInIt:
    """Every family and scenario pair must actually produce cases.

    The first version of this generator produced none at all for a clear camera
    drift, for a reason that looked like a sensible guard: a case should be loud on
    the channel its fault moves and quiet on the rest. But a camera moves the
    MEASUREMENT, so it shows up on the reference and as an apparent shift of the
    whole seam, and that second channel is the entire confound this project spent a
    study on. The guard threw every camera case away, the report said "8 pairs"
    regardless, and the published scores were measured on a test with a hole in it.
    """

    def test_every_planned_pair_builds(self):
        from scripts.eval_hard_drift import PLAN

        for family, scenario in PLAN:
            got = [drift.build_hard_case(family, scenario, seed, parts=30)
                   for seed in (3, 4, 5, 6)]
            made = [g for g in got if g is not None]
            assert made, f"{family} + {scenario} produced no cases on any seed"

    def test_a_camera_is_allowed_to_move_the_seam_as_well_as_the_reference(self):
        """The thing the old guard got wrong, stated as a property."""
        rows, _ = drift.build_stream(30, "camera", 4321, 2.0, reference=True)
        d = drift.discriminators(rows)
        assert d["reference_ratio"] > drift.VISIBLE_RATIO
        assert d["shift_ratio"] > drift.VISIBLE_RATIO, (
            "a drifting camera moves the apparent seam too; if this ever stops "
            "being true the confound study has no subject")
        assert drift._quiet_elsewhere("camera", d, 2.0), (
            "both loud channels are expected for a camera, so it must still count "
            "as quiet elsewhere")

    def test_a_press_that_also_shifts_is_rejected(self):
        """The guard must still do its job on the scenarios it was written for."""
        loud_everywhere = {"tilt_ratio": 9.0, "shift_ratio": 9.0, "reference_ratio": 0.2}
        assert not drift._quiet_elsewhere("press", loud_everywhere, 2.0)

    def test_cases_land_inside_the_band_they_were_asked_for(self):
        lo, hi = drift.HARD_BANDS["marginal"]
        for seed in (3, 4, 5):
            got = drift.build_hard_case("marginal", "press", seed, parts=30)
            if got is None:
                continue
            _, _, d = got
            assert lo <= d["tilt_ratio"] <= hi

    def test_a_marginal_fault_does_not_clear_the_threshold(self):
        """Otherwise it is not marginal, it is just a fault."""
        for seed in (3, 4, 5, 6):
            got = drift.build_hard_case("marginal", "fixture", seed, parts=30)
            if got is None:
                continue
            _, _, d = got
            assert d["shift_ratio"] < drift.VISIBLE_RATIO + 0.4
