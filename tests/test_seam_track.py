"""Tests for seam tracking.

Seams are synthesised with a known shape and a known gap, so the measurement can
be scored. The cases that matter are the ones a taught program gets wrong: a seam
that curves away from the straight line between its ends, and a gap that opens
over part of the length.
"""

from __future__ import annotations

import numpy as np
import pytest

from cell.seam_track import TOUCHING_TOLERANCE_MM, track_seam

SEAM_START = np.array([0.0, 0.0, 0.00])
SEAM_END = np.array([0.0, 0.0, 0.40])
SPACING = 0.0004


def build_corner(
    bow_mm: float = 0.0,
    gap_mm_at=lambda z: 0.0,
    face_extent: float = 0.06,
    noise_mm: float = 0.0,
) -> np.ndarray:
    """Two folded faces meeting at a vertical seam, as a point cloud.

    Face A lies in the x = seam plane, face B in the y = seam plane, so they meet
    at right angles. A bow displaces the seam sideways along its length, the way
    camber or springback would. A gap pulls each face back from the seam.
    """
    rng = np.random.default_rng(0)
    points = []
    for z in np.arange(SEAM_START[2], SEAM_END[2], SPACING):
        t = (z - SEAM_START[2]) / (SEAM_END[2] - SEAM_START[2])
        # A bow: zero at both ends, worst in the middle.
        bow = (bow_mm / 1000.0) * np.sin(np.pi * t)
        seam_x, seam_y = bow, 0.0
        back = gap_mm_at(z) / 2000.0  # each face retreats half the gap

        for offset in np.arange(back, face_extent, SPACING):
            # Range noise is across each face, not along it, so the edge of the
            # material stays sharp however noisy the depth is.
            ja = rng.normal(0, noise_mm / 1000.0) if noise_mm else 0.0
            jb = rng.normal(0, noise_mm / 1000.0) if noise_mm else 0.0
            points.append([seam_x + ja, seam_y - offset, z])   # face A, along -y
            points.append([seam_x + offset, seam_y + jb, z])   # face B, along +x
    return np.array(points)


# ---------------------------------------------------------------- basic geometry


def test_finds_a_straight_seam_where_it_is():
    track = track_seam(build_corner(), SEAM_START, SEAM_END, stations=12)
    assert len(track.measured) >= 10
    path = track.path()
    assert np.abs(path[:, 0]).max() < 0.001
    assert np.abs(path[:, 1]).max() < 0.001


def test_touching_edges_read_as_no_gap():
    track = track_seam(build_corner(), SEAM_START, SEAM_END, stations=12)
    assert track.gaps_mm().max() < TOUCHING_TOLERANCE_MM
    assert all(s.touching for s in track.measured)


def test_a_uniform_gap_is_measured():
    track = track_seam(
        build_corner(gap_mm_at=lambda z: 1.6), SEAM_START, SEAM_END, stations=12
    )
    gaps = track.gaps_mm()
    assert gaps.mean() == pytest.approx(1.6, abs=0.4)
    assert not any(s.touching for s in track.measured)


def test_a_gap_that_opens_over_part_of_the_length():
    """The press check: edges touching at one end, opening towards the other."""
    track = track_seam(
        build_corner(gap_mm_at=lambda z: 0.0 if z < 0.22 else 2.2),
        SEAM_START,
        SEAM_END,
        stations=16,
    )
    summary = track.summary()
    assert summary["gap_mm_max"] > 1.5
    assert summary["stations_touching"] > 3
    assert summary["gap_worst_at_mm"] > 200


# --------------------------------------------------- the thing a taught path misses


def test_a_bowed_seam_is_followed_not_straightened():
    """A straight taught path is exactly what springback and camber defeat.

    The measured path has to bend with the seam. If it comes back straight the
    measurement is useless, however tidy its numbers look.
    """
    track = track_seam(build_corner(bow_mm=3.0), SEAM_START, SEAM_END, stations=20)
    path = track.path()

    middle = path[len(path) // 2]
    assert middle[0] * 1000.0 == pytest.approx(3.0, abs=0.8), (
        "the seam bows 3 mm at its midpoint and the measurement must show it"
    )
    assert track.wander_mm() == pytest.approx(3.0, abs=0.8)


def test_a_straight_seam_reports_no_wander():
    track = track_seam(build_corner(), SEAM_START, SEAM_END, stations=20)
    assert track.wander_mm() < 0.5


def test_wander_grows_with_the_bow():
    small = track_seam(build_corner(bow_mm=1.0), SEAM_START, SEAM_END, stations=20)
    large = track_seam(build_corner(bow_mm=4.0), SEAM_START, SEAM_END, stations=20)
    assert large.wander_mm() > small.wander_mm() + 1.5


# ------------------------------------------------------------- looking ahead


def test_measurement_leads_the_torch():
    track = track_seam(build_corner(), SEAM_START, SEAM_END, stations=20)
    ahead = track.ahead_of(along_m=0.10, lead_m=0.05)
    assert ahead is not None
    assert ahead.along_m >= 0.15


def test_nothing_ahead_at_the_end_of_the_seam():
    track = track_seam(build_corner(), SEAM_START, SEAM_END, stations=20)
    assert track.ahead_of(along_m=0.39, lead_m=0.05) is None


# ------------------------------------------------------------------ robustness


def test_a_slice_with_only_one_face_is_refused_not_guessed():
    points = build_corner()
    points = points[~((points[:, 1] < -0.001) & (points[:, 2] > 0.30))]
    track = track_seam(points, SEAM_START, SEAM_END, stations=16)
    refused = [s for s in track.stations if not s.valid]
    assert refused, "a slice showing one face must be refused, not reported"
    assert any("one face" in (s.reason or "") for s in refused)


def test_summary_survives_a_seam_nothing_could_be_measured_on():
    track = track_seam(np.zeros((5, 3)), SEAM_START, SEAM_END, stations=6)
    assert track.summary()["stations_measured"] == 0


def test_rejects_a_zero_length_seam():
    with pytest.raises(ValueError):
        track_seam(build_corner(), SEAM_START, SEAM_START)


def test_survives_sensor_noise():
    points = build_corner(bow_mm=2.0, noise_mm=0.4)
    track = track_seam(points, SEAM_START, SEAM_END, stations=16)
    assert len(track.measured) >= 12
    assert track.wander_mm() == pytest.approx(2.0, abs=1.2)
