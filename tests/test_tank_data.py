"""The tank numbers the README quotes, recomputed from the recorded data.

tests/test_tank.py pins the scoring rules on made-up rows and
tests/test_tank_truth.py pins the answer key. This file pins what those rules
produce on the recorded run: the survey marks, the probe arms, the chance band,
the paired tests and the layout and tie-break studies. Every expected number is
one the scripts print (score_survey.py --replay, significance.py, score_probes.py
--baselines and --tiebreaks, layout_sweep.py), so a change to any of them fails
here before it reaches a page that quotes the old one.

Read only. Nothing here writes under tank/data or reaches the network, and the
fixture below makes every way into the model client raise, so a test that
wandered into a paid call would fail instead of spending credit.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from brain import nebius
from service import tank_view
from tank import layout_sweep, score_probes, score_survey, significance
from tank.probe_planner import Memory, sweep_rule
from tank.thickness import thickness_mm
from tank.truth import load_answers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tank" / "data"
SURVEY = DATA / "survey"
ARCHIVE = SURVEY / "archive_four_point"
MANIFEST = json.loads((SURVEY / "manifest.json").read_text(encoding="utf-8"))
SCENE = json.loads((DATA / "tank_scene.json").read_text(encoding="utf-8"))
FRAMES = MANIFEST["frames"]
ANSWERS = load_answers(SURVEY / "inspection.jsonl")
BUDGET = 2


def _no_model(*args, **kwargs):
    raise AssertionError("a tank data test tried to reach a model")


@pytest.fixture(autouse=True)
def no_paid_calls(monkeypatch):
    """A key may well be in .env. Nothing in this file is allowed to use it."""
    import openai

    monkeypatch.setattr(nebius, "chat", _no_model)
    monkeypatch.setattr(nebius, "client", _no_model)
    monkeypatch.setattr(nebius, "OpenAI", _no_model)
    monkeypatch.setattr(openai, "OpenAI", _no_model)


def test_the_guard_is_on():
    """The scorers reach the client through this one module, so this is the
    whole of the door."""
    assert score_probes.nebius is nebius and score_survey.nebius is nebius
    with pytest.raises(AssertionError):
        nebius.chat([{"role": "user", "content": "hello"}])
    with pytest.raises(AssertionError):
        nebius.client()


def _rows():
    return score_survey.replayed_rows(FRAMES, ANSWERS)


def _arm(chooser):
    return score_probes.run(FRAMES, ANSWERS, chooser, BUDGET, "")


def _sweep():
    return _arm(lambda o, s, m, b, f, r: sweep_rule(o, s, m, b))


def _found(got):
    return [(s, round(o, 2)) for s, o, _ in got["rejectable_found"]]


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


# ------------------------------------------------------------- the survey


def test_the_replay_gives_the_published_survey_marks():
    """score_survey.py --replay, DeepSeek V4.1 Flash, 24 frames."""
    got = score_survey.score(_rows())
    assert (got["frames"], got["judged"], got["declined"]) == (24, 24, 0)
    assert got["pass_or_defect_right"] == 23
    assert got["gate_right_for_a_fault_in_view"] == 22
    assert (got["faults_found"], got["faults_present"]) == (10, 15)
    assert got["invented"] == 0
    assert got["named_right"] == 19
    assert got["frames_all_found"] == 19
    assert (got["missed"], got["false_alarms"], got["called_a_sliver"]) == (1, 0, 0)


def test_the_one_miss_is_the_crack_at_station_12():
    missed = [r["station"] for r in _rows()
              if r["truth"] != "none" and r["called"] == "none"]
    assert missed == [12]


def test_four_of_the_five_faults_on_the_run_were_named_somewhere():
    faults = score_survey.per_fault(FRAMES, _rows())
    assert len(faults) == 5
    assert sum(1 for f in faults if f["named_in"]) == 4
    never = [(f["id"], f["kind"], f["frames"]) for f in faults if not f["named_in"]]
    assert never == [("D001", "undercut", [1, 2, 3, 4])]
    assert sum(len(f["frames"]) for f in faults) == 15


def test_the_replay_prints_every_key_disagreement(monkeypatch, capsys):
    """Station 2 was recorded against a key that did not count the rust in
    it; the key now does. The replay must say so, and say nothing else."""
    monkeypatch.setattr(sys, "argv", ["score_survey.py", "--replay"])
    assert score_survey.main() == 0
    out = capsys.readouterr().out
    assert "pass or defect 23/24" in out
    assert "faults found 10/15" in out
    assert "differs from the key now, at 1 frame(s)" in out
    assert "station 2: recorded against undercut ['undercut']" in out


def test_without_a_flag_the_survey_scorer_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["score_survey.py"])
    assert score_survey.main() == 2
    assert "nothing run" in capsys.readouterr().out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_live_run_refuses_to_write_over_the_recorded_verdicts(
        monkeypatch, capsys, tmp_path):
    """Only the recorded-verdicts check may stop this run.

    The target is a copy standing in for the recorded file, and --force is
    passed, so the generic "already exists" refusal cannot step in for it. If
    the check were removed the run would get as far as the model, which the
    fixture makes raise, and only the copy would be truncated.
    """
    recorded = SURVEY / "inspection.jsonl"
    before = _sha(recorded)
    copy = tmp_path / "inspection.jsonl"
    copy.write_bytes(recorded.read_bytes())
    monkeypatch.setattr(score_survey, "RECORDED", copy)
    monkeypatch.setenv("NEBIUS_API_KEY", "dummy-not-a-key")
    monkeypatch.setattr(sys, "argv", [
        "score_survey.py", "--live", "--force",
        "--answers", str(recorded), "--out", str(copy)])
    assert score_survey.main() == 1
    assert "over the recorded verdicts" in capsys.readouterr().out
    assert _sha(copy) == before
    assert _sha(recorded) == before


# ------------------------------------------------------- significance.py


def test_significance_reads_the_same_marks_as_the_scorer(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["significance.py", "--json"])
    assert significance.main() == 0
    numbers = json.loads(capsys.readouterr().out)
    ins = numbers["inspection"]
    got = score_survey.score(_rows())
    m = ins["measures"]
    assert (m["pass_or_defect"]["right"], m["pass_or_defect"]["of"]) == (
        got["pass_or_defect_right"], got["judged"])
    assert m["right_call_for_a_fault_in_view"]["right"] == got[
        "gate_right_for_a_fault_in_view"]
    assert (m["faults_found"]["right"], m["faults_found"]["of"]) == (
        got["faults_found"], got["faults_present"])
    assert m["headline_named"]["right"] == got["named_right"]
    assert m["frames_with_every_fault_found"]["right"] == got["frames_all_found"]
    assert (ins["invented"], ins["missed"], ins["false_alarms"]) == (
        got["invented"], got["missed"], got["false_alarms"])
    assert ins["missed_stations"] == [12]
    assert (ins["faults_on_the_run"]["right"], ins["faults_on_the_run"]["of"]) == (4, 5)
    # The one sided 95 percent exact lower bounds the README quotes.
    assert m["pass_or_defect"]["lower_95_one_sided"] == pytest.approx(0.817, abs=5e-4)
    assert m["right_call_for_a_fault_in_view"]["lower_95_one_sided"] == pytest.approx(
        0.760, abs=5e-4)
    assert m["faults_found"]["lower_95_one_sided"] == pytest.approx(0.423, abs=5e-4)
    assert m["headline_named"]["lower_95_one_sided"] == pytest.approx(0.611, abs=5e-4)
    assert ins["faults_on_the_run"]["lower_95_one_sided"] == pytest.approx(0.343, abs=5e-4)
    # Its probing section agrees with the arms recomputed here.
    probe = numbers["probing"]
    assert probe["rule"]["found"] == len(_sweep()["rejectable_found"]) == 16
    assert probe["chance"]["mean"] == pytest.approx(8.0)
    assert probe["rule_ahead_in_every_run"] is True
    assert probe["model_ahead_in_any_run"] is False


def test_the_paired_tests_are_one_per_run_and_never_pooled():
    """Exact McNemar, the sweep against each recorded Nemotron 3 Nano run."""
    probe = significance.probing(MANIFEST)
    got = [(r["file"], r["found"], r["only_model"], r["only_rule"], r["mcnemar_p"])
           for r in probe["runs"]]
    assert got == [
        ("probing.json", 9, 5, 12, pytest.approx(0.1435, abs=1e-4)),
        ("probing_power.json", 12, 3, 7, pytest.approx(0.3438, abs=1e-4)),
        ("probing_power.json", 10, 3, 9, pytest.approx(0.1460, abs=1e-4)),
        ("probing_blind.json", 11, 4, 9, pytest.approx(0.2668, abs=1e-4)),
    ]
    assert significance.mcnemar_exact(5, 12) == pytest.approx(0.1435, abs=1e-4)
    assert not any(r["mcnemar_p"] < 0.05 for r in probe["runs"])
    assert probe["skipped"] == []


def test_each_model_run_reached_four_of_the_five_patches():
    probe = significance.probing(MANIFEST)
    assert probe["hidden_patches"] == ["D006", "D007", "D008", "D009", "D010"]
    assert len(probe["rule"]["patches_found"]) == 5
    assert [r["patches_missed"] for r in probe["runs"]] == [
        ["D009"], ["D008"], ["D010"], ["D010"]]


# ------------------------------------------------------- the probe arms


def test_the_sweep_finds_16_of_28_with_48_measurements():
    got = _sweep()
    assert got["spent"] == 48
    assert len(got["rejectable_found"]) == 16
    assert got["patches_found"] == ["D006", "D007", "D008", "D009", "D010"]
    assert _found(got) == [
        (2, -0.30), (3, -0.30), (3, -0.45), (8, 0.30), (9, 0.30), (9, 0.15),
        (10, 0.30), (13, -0.45), (13, -0.30), (14, -0.30), (14, -0.45),
        (15, -0.45), (15, -0.30), (16, -0.30), (16, -0.45), (20, 0.45)]


def test_the_sweep_given_only_the_models_view_finds_11():
    """Matched on information, the rule is level with the model's 9 to 12."""
    got = _arm(lambda o, s, m, b, f, r: score_probes.sweep_models_view(o, s, m, b))
    assert got["spent"] == 48
    assert len(got["rejectable_found"]) == 11
    assert len(got["patches_found"]) == 5


def test_on_the_seam_and_probe_beside_rust_each_find_2():
    seam = _arm(lambda o, s, m, b, f, r: score_probes.on_the_seam(o, s, m, b))
    beside = _arm(lambda o, s, m, b, f, r: score_probes.where_it_saw(o, s, m, b, r))
    for got in (seam, beside):
        assert got["spent"] == 48
        assert _found(got) == [(2, -0.15), (3, -0.15)]
        assert got["patches_found"] == ["D007"]


def test_probe_beside_rust_ignores_left_and_right_and_probes_the_seam_for_a_crack():
    """The reported side runs along the seam, not across it, so it cannot pick
    an offset; and the probe measures wall loss, so a weld fault gets the seam."""
    offsets = [c["offset_m"] for c in FRAMES[0]["probe_candidates"]]
    memory = Memory()

    def pick(row):
        return score_probes.where_it_saw(offsets, 5, memory, BUDGET, row)

    left = pick({"called": "corrosion", "faults": ["corrosion"], "where": "left"})
    right = pick({"called": "corrosion", "faults": ["corrosion"], "where": "right"})
    assert left == right == [-0.15, 0.15]
    crack = pick({"called": "crack", "faults": ["crack"], "where": "left"})
    assert crack == score_probes.on_the_seam(offsets, 5, memory, BUDGET)
    assert crack[0] == 0.0


def test_the_chance_band_is_an_exact_distribution():
    chance = score_probes.chance_distribution(FRAMES, BUDGET)
    assert sum(chance["pmf"]) == pytest.approx(1.0, abs=1e-12)
    assert chance["mean"] == pytest.approx(8.0)
    assert chance["band_95"] == [4, 12]
    assert score_probes.p_at_least(chance, 16) == pytest.approx(0.0002, abs=5e-5)
    assert score_probes.p_at_least(chance, 11) == pytest.approx(0.1135, abs=5e-5)


def test_the_oracle_finds_23():
    assert score_probes.oracle_bound(FRAMES, BUDGET, ANSWERS)["found"] == 23


def test_the_sweeps_16_is_near_the_top_of_its_tie_break_orders():
    rejectable = sum(1 for f in FRAMES for c in f["probe_candidates"] if c["rejectable"])
    study = score_probes.tiebreak_study(FRAMES, ANSWERS, BUDGET, rejectable)
    assert study["orders"] == 5040
    assert (study["min"], study["max"], study["median"]) == (7, 18, 12)
    assert study["mean"] == pytest.approx(12.35, abs=5e-3)
    assert study["shipped_found"] == 16
    assert study["share_below_shipped"] == pytest.approx(0.939, abs=5e-4)
    view = study["models_view"]
    assert (view["min"], view["max"], view["median"]) == (7, 19, 13)
    assert view["mean"] == pytest.approx(12.76, abs=5e-3)


def test_the_hand_placed_layout_favours_the_sweep(monkeypatch, capsys):
    """1000 random layouts of the same five patches, seed 1."""
    monkeypatch.setattr(sys, "argv", ["layout_sweep.py"])
    assert layout_sweep.main() == 0
    out = capsys.readouterr().out
    assert "reproduces the manifest and probing.json: yes" in out
    sweep_line = next(line for line in out.splitlines() if line.strip().startswith("sweep "))
    assert "57.1%" in sweep_line and "40.7%" in sweep_line
    assert "94.9% of layouts below" in sweep_line
    seam_line = next(line for line in out.splitlines()
                     if line.strip().startswith("on the seam "))
    assert "1.2% of layouts below" in seam_line
    assert "above the chance mean in 907 of 1000 layouts, above on the seam in 572" in out


def test_the_layout_studys_copy_of_the_sweep_is_the_sweep():
    """layout_sweep cannot import the planner (it imports the model client), so
    it carries its own copy; on the shipped layout the two must pick alike."""
    limit = MANIFEST["acceptance"]["thinning"]["limit_fraction"]
    stations = layout_sweep.answers(MANIFEST, SCENE["defects"], limit)
    copy = layout_sweep.sweep(stations, BUDGET)
    shipped = [(s, o) for s, picked in _sweep()["rows"] for o in picked]
    assert copy == shipped


# --------------------------------------------------- the recorded data


def test_every_probe_answer_is_the_thickness_model_at_that_point():
    limit = MANIFEST["acceptance"]["thinning"]["limit_fraction"]
    points = [c for f in FRAMES for c in f["probe_candidates"]]
    assert len(points) == 168
    for c in points:
        left, took = thickness_mm(MANIFEST["defects"], c["theta_rad"], c["z_m"])
        assert left == pytest.approx(c["true_thickness_mm"], abs=1e-9)
        assert took == c["thinned_by"]
        assert c["remaining_fraction"] == round(left / c["nominal_mm"], 4)
        assert c["rejectable"] == (c["remaining_fraction"] < limit)
    assert sum(1 for c in points if c["rejectable"]) == 28
    assert sum(1 for c in points if c["thinned_by"]) == 82


def test_every_thinned_point_names_a_patch_of_wall_loss_in_the_scene():
    kinds = {d["id"]: d["kind"] for d in SCENE["defects"]}
    for f in FRAMES:
        for c in f["probe_candidates"]:
            for patch in c["thinned_by"]:
                assert kinds.get(patch) == "thinning", (f["station"], patch)


def test_every_failing_point_is_on_a_patch_no_camera_can_see():
    visible = {d["id"] for d in SCENE["defects"] if d["visible"]}
    failing = [c for f in FRAMES for c in f["probe_candidates"] if c["rejectable"]]
    assert failing
    for c in failing:
        assert not visible & set(c["thinned_by"])


def test_every_current_probe_record_is_on_the_28_point_scene():
    records = [p for p in SURVEY.rglob("probing*.json") if ARCHIVE not in p.parents]
    assert {p.name for p in records} >= {"probing.json", "probing_power.json",
                                         "probing_blind.json", "probing_baselines.json"}
    for p in records:
        assert json.loads(p.read_text(encoding="utf-8"))["rejectable"] == 28, p.name


def test_no_code_reads_the_archived_four_point_files():
    archived = {p.name for p in ARCHIVE.glob("*.json")}
    assert archived
    skip = {".venv", ".git", "node_modules", "__pycache__"}
    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".js", ".html"} or not path.is_file():
            continue
        if skip & set(path.relative_to(ROOT).parts) or path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name in archived:
            assert name not in text, f"{path.relative_to(ROOT)} names {name}"
    for name in (*significance.PROBE_RUNS, *tank_view.PROBING):
        assert (SURVEY / name).parent == SURVEY


# ------------------------------------------------------- the dashboard


def test_the_dashboard_counts_are_the_scorers():
    view = tank_view.summary()
    assert view["available"] and view["scored"]
    got = score_survey.score(_rows())
    counts, gate = view["counts"], view["gate"]
    assert counts["right"] == got["named_right"]
    assert counts["missed"] == got["missed"]
    assert counts["misnamed"] == got["misnamed"]
    assert counts["false_alarm"] == got["false_alarms"]
    assert counts["sliver"] == got["called_a_sliver"]
    assert counts["declined"] == got["declined"]
    assert counts["unjudged"] == 0
    assert gate["judged"] == got["judged"]
    assert gate["right"] == got["pass_or_defect_right"]
    assert gate["strict_right"] == got["gate_right_for_a_fault_in_view"]
    assert gate["named_a_lesser_fault_there"] == got["named_a_lesser_fault_that_was_there"]
    faults = score_survey.per_fault(FRAMES, _rows())
    assert view["faults_on_the_run"]["total"] == len(faults)
    assert view["faults_on_the_run"]["named"] == sum(1 for f in faults if f["named_in"])
    probe = view["probe"]
    assert (probe["points"], probe["thinned"], probe["rejectable"]) == (168, 82, 28)
    assert probe["rejectable_under_a_pass"] == 20
    assert probe["rejectable_under_a_clean_frame"] == 18
    assert probe["rejectable_on_a_visible_patch"] == 0
    assert probe["best_possible"] == 23


# ------------------------------------------------- nothing gets spent


def test_the_probe_scorer_writes_nothing_and_asks_nothing_without_live(monkeypatch, capsys):
    """A key in the environment must not turn a plain run into a paid one."""
    monkeypatch.setenv("NEBIUS_API_KEY", "dummy-not-a-key")
    monkeypatch.setattr(sys, "argv", ["score_probes.py"])
    before = _tree(DATA)
    assert score_probes.main() == 0
    assert _tree(DATA) == before
    out = capsys.readouterr().out
    assert "Nothing written" in out
    assert "20 are under frames the inspector passed, 18 under frames" in out
