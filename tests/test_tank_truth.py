"""The tank's answer key: that it can be rebuilt without Blender, and means what the pictures show.

Every inspection number in the README is a verdict marked against this key, so
these tests pin the key itself rather than any score. If one fails, the scores
are not wrong yet, but nobody can say they are right.
"""

import hashlib
import json
from pathlib import Path

import pytest

from tank import truth

ROOT = Path(__file__).resolve().parent.parent
SURVEY = ROOT / "tank" / "data" / "survey"
MANIFEST = json.loads((SURVEY / "manifest.json").read_text(encoding="utf-8"))
SCENE = json.loads((ROOT / "tank" / "data" / "tank_scene.json").read_text(encoding="utf-8"))


def test_the_camera_projects_exactly_as_blender_does():
    """630 wall points, six stations, Blender's own numbers, recorded once."""
    check = json.loads((SURVEY / "projection_check.json").read_text(encoding="utf-8"))
    worst = 0.0
    count = 0
    for station in check["stations"]:
        for theta, z, inset, x, y, depth in station["points"]:
            px, py, pd = truth.project(truth.wall_point(theta, z, inset),
                                       station["theta_rad"], station["z_m"])
            worst = max(worst, abs(px - x) * truth.RESOLUTION[0],
                        abs(py - y) * truth.RESOLUTION[1])
            assert pd == pytest.approx(depth, abs=1e-5)
            count += 1
    assert count == 630
    # Blender works in single precision; a twentieth of a pixel is that, not a
    # different camera. A wrong sensor or lens would be off by tens of pixels.
    assert worst < 0.05


def test_the_key_in_the_manifest_is_the_key_this_code_computes():
    """Nobody has to trust a frozen file: the key is rebuilt from the scene."""
    for frame in MANIFEST["frames"]:
        again = truth.faults_in_view(SCENE["defects"], frame["theta_rad"], frame["z_m"])
        assert again == frame["faults_in_view"], f"station {frame['station']}"


def test_the_manifest_carries_the_scene_it_was_scored_against():
    assert MANIFEST["defects"] == SCENE["defects"]


def test_every_seam_fault_on_the_run_is_projected_from_its_rendered_vertices():
    drawn = truth.drawn_geometry()
    for fid in MANIFEST["survey_run"]["faults_on_it"]:
        assert len(drawn.get(fid, ())) >= 20, fid
    for frame in MANIFEST["frames"]:
        for f in frame["faults_in_view"]:
            if f["id"] in MANIFEST["survey_run"]["faults_on_it"]:
                assert f["footprint"] == "drawn"


def test_station_12_holds_the_crack_frame_11_named():
    """The frame the old key called clean. The crack is drawn a third of the way
    up the bead, and projected at the crown it fell just outside the picture."""
    by_station = {f["station"]: f for f in MANIFEST["frames"]}
    for station in (11, 12):
        worst, seen, _ = truth.truth_for(by_station[station])
        assert worst == "crack"
        ids = {f["id"] for f in by_station[station]["faults_in_view"]}
        assert "D003" in ids


def test_the_recorded_verdicts_were_scored_against_this_key():
    """The truth stored beside each recorded verdict, at the time it was
    recorded, equals the key as it stands. If this fails, the replay prints
    the disagreement; it must never be resolved by overwriting either side."""
    answers = truth.load_answers(SURVEY / "inspection.jsonl")
    assert len(answers) == len(MANIFEST["frames"])
    for frame in MANIFEST["frames"]:
        row = answers[frame["station"]]
        worst, seen, _ = truth.truth_for(frame)
        assert row["truth"] == worst, f"station {frame['station']}"
        assert sorted(row["in_view"]) == sorted(seen), f"station {frame['station']}"


def test_the_frames_are_the_pixels_the_verdicts_were_given_against():
    recorded = {}
    for line in (SURVEY / "frames.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            recorded[name.strip().lstrip("*")] = digest
    assert len(recorded) == len(MANIFEST["frames"])
    for frame in MANIFEST["frames"]:
        data = (SURVEY / frame["rgb"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == recorded[frame["rgb"]], frame["rgb"]


def test_hidden_wall_loss_never_counts_against_the_camera():
    for frame in MANIFEST["frames"]:
        worst, seen, grazed = truth.truth_for(frame)
        hidden = [f for f in frame["faults_in_view"] if not f["visible_to_a_camera"]]
        for f in hidden:
            assert f["kind"] == "thinning"
        if not [f for f in frame["faults_in_view"] if f["visible_to_a_camera"]]:
            assert (worst, seen, grazed) == ("none", [], [])


def test_the_camera_mount_is_read_from_the_scene_builder():
    from tank.adhesion import scene_constants

    built = scene_constants("CAMERA_EYE", "CAMERA_LENS_MM", "RADIUS", "BEAD_R")
    used = {"CAMERA_EYE": truth.CAMERA_EYE, "CAMERA_LENS_MM": truth.LENS_MM,
            "RADIUS": truth.RADIUS, "BEAD_R": truth.BEAD_R}
    assert used == built
    assert MANIFEST["camera"]["camera_standoff_m"] == built["CAMERA_EYE"][2]


def test_a_second_verdict_for_one_station_is_refused(tmp_path):
    rows = [{"model": "m", "station": 3, "verdict": "pass", "called": "none"},
            {"model": "m", "station": 3, "verdict": "defect", "called": "crack"}]
    path = tmp_path / "answers.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError):
        truth.load_answers(path)


def test_worst_follows_the_order_the_inspector_is_given():
    from tank import inspect_seam

    assert inspect_seam.SEAM_DEFECTS == truth.SEVERITY
    assert truth.worst(["corrosion", "undercut"]) == "undercut"
    assert truth.worst([]) == "none"
