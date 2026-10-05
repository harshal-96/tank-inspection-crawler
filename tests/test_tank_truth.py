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


# Stations where the key has moved since the verdict was recorded, and why. The
# recorded rows are evidence and are never rewritten to match; the replay
# prints each of these, and anything not listed here is a failure.
KEY_MOVED_SINCE_RECORDING = {
    # The old key measured the frame on a grid round the crawler's origin,
    # 0.46 m behind the lens, so it saw only the left sixth of the picture and
    # called the rust filling the right edge of frame 2 a sliver.
    2: {"recorded": ("undercut", ["undercut"]),
        "now": ("undercut", ["corrosion", "undercut"])},
}


def test_the_recorded_verdicts_were_scored_against_this_key():
    """The truth stored beside each recorded verdict, at the time it was
    recorded, equals the key as it stands, except where it is listed above.
    It must never be resolved by overwriting either side."""
    answers = truth.load_answers(SURVEY / "inspection.jsonl")
    assert len(answers) == len(MANIFEST["frames"])
    for frame in MANIFEST["frames"]:
        station = frame["station"]
        row = answers[station]
        worst, seen, _ = truth.truth_for(frame)
        stored = (row["truth"], sorted(row["in_view"]))
        now = (worst, sorted(seen))
        if station in KEY_MOVED_SINCE_RECORDING:
            moved = KEY_MOVED_SINCE_RECORDING[station]
            assert stored == (moved["recorded"][0], sorted(moved["recorded"][1]))
            assert now == (moved["now"][0], sorted(moved["now"][1]))
        else:
            assert stored == now, f"station {station}"


def test_the_frame_is_sampled_where_the_lens_is_looking():
    """The share of the picture a fault fills is measured across the whole
    picture. The grid that sat round the crawler's origin covered 33 points
    at its left edge; this one lands every sample inside the frame and spans
    it from edge to edge."""
    frame = MANIFEST["frames"][12]
    theta, z = frame["theta_rad"], frame["z_m"]
    samples = truth.frame_samples(theta, z, truth.RUST_INSET)
    cols, rows = truth.FRAME_SAMPLES
    assert len(samples) == cols * rows
    along = [(t - theta) * truth.RADIUS for t, _ in samples]
    assert min(along) < 0.32 and max(along) > 0.58
    for t, y in samples[:: 97]:
        x, yy, d = truth.project(truth.wall_point(t, y, truth.RUST_INSET), theta, z)
        assert truth.in_frame((x, yy, d))


def test_the_visible_rust_in_the_key_is_the_rust_in_the_pixels():
    """An independent check against the rendered frames. Rust is the only warm,
    saturated colour in the scene, so its share of the pixels is a second
    measurement of how much of the picture it fills. It runs a little under
    the key, because pits and shadowed rust are not orange, but the two must
    agree on which side of the counting threshold every frame falls."""
    import numpy as np
    from PIL import Image

    for frame in MANIFEST["frames"]:
        a = np.asarray(Image.open(SURVEY / frame["rgb"]).convert("RGB"), dtype=float)
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        saturation = (a.max(axis=2) - a.min(axis=2)) / np.maximum(a.max(axis=2), 1)
        pixels = float(((r > g) & (g > b) & (saturation > 0.35) & (r - b > 40)).mean())
        key = next((f["of_the_frame"] for f in frame["faults_in_view"]
                    if f["kind"] == "thinning" and f["visible_to_a_camera"]), 0.0)
        assert (key >= truth.TRACE) == (pixels >= truth.TRACE), frame["station"]
        assert abs(key - pixels) < 0.25, frame["station"]


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
