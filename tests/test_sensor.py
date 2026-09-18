"""The sensor model, and the path estimate that has to survive it.

These exist because the robustness study is only worth something if the thing it
degrades actually degrades the way a depth camera does, and if the estimator it
scores is doing what its docstring claims. Both were wrong at some point:

  - the inlier band for the face fits defaulted wider than the sheet is thick,
    which let one plane claim a panel's inner AND outer surface and put its
    offset between them. It biased the seam by more than a millimetre and made
    the better estimator look worse than the one it replaced.
"""

from __future__ import annotations

import numpy as np
import pytest

from cell import geometry as geo
from cell.sensor import CLEAN, HARSH_SPECULAR, TYPICAL_STEREO, DepthSensor


def flat_frame(value: float = 1.0, shape=(120, 160)) -> np.ndarray:
    return np.full(shape, value, dtype=float)


def test_a_clean_sensor_changes_nothing():
    """The zero arm of a sweep must go through the same code as every other arm."""
    frame = flat_frame()
    assert np.array_equal(CLEAN.degrade(frame), frame)


def test_range_noise_grows_with_the_square_of_distance():
    """Triangulation error is not linear in range, and quoting it as such misleads."""
    sensor = DepthSensor(axial_mm_at_1m=2.0)
    rng = np.random.default_rng(0)

    near = sensor.degrade(flat_frame(1.0, (400, 400)), rng)
    far = sensor.degrade(flat_frame(2.0, (400, 400)), rng)

    spread_near = float(np.std(near - 1.0))
    spread_far = float(np.std(far - 2.0))

    assert spread_near == pytest.approx(0.002, rel=0.15)
    # Twice the distance is four times the error, not twice.
    assert spread_far / spread_near == pytest.approx(4.0, rel=0.2)


def test_dropout_hits_the_fraction_asked_for():
    """Whatever the patch size, the requested fraction is what goes missing."""
    for patch in (1, 4, 8):
        sensor = DepthSensor(dropout=0.25, dropout_patch_px=patch)
        out = sensor.degrade(flat_frame(1.0, (400, 400)), np.random.default_rng(1))
        lost = float(np.mean(~np.isfinite(out)))
        assert lost == pytest.approx(0.25, abs=0.06), f"patch {patch} lost {lost:.3f}"


def test_dropout_comes_in_patches_rather_than_speckle():
    """Specular loss on sheet metal is regional. Speckle is the wrong model."""
    speckle = DepthSensor(dropout=0.2, dropout_patch_px=1).degrade(
        flat_frame(1.0, (300, 300)), np.random.default_rng(2))
    patchy = DepthSensor(dropout=0.2, dropout_patch_px=7).degrade(
        flat_frame(1.0, (300, 300)), np.random.default_rng(2))

    def neighbours_also_lost(frame):
        lost = ~np.isfinite(frame)
        return float(np.mean(lost[:, :-1] & lost[:, 1:]) / max(np.mean(lost), 1e-9))

    assert neighbours_also_lost(patchy) > neighbours_also_lost(speckle) * 2


def test_flying_pixels_only_appear_at_a_depth_step():
    """A flat surface has no discontinuity, so it should gain no flying pixels."""
    flat = DepthSensor(flying_pixels=1.0).degrade(flat_frame(1.0), np.random.default_rng(3))
    assert np.allclose(flat, 1.0)

    stepped = flat_frame(1.0, (100, 100))
    stepped[:, 50:] = 1.1
    out = DepthSensor(flying_pixels=1.0).degrade(stepped, np.random.default_rng(3))

    moved = ~np.isclose(out, stepped)
    assert moved.any()
    # Everything that moved sits near the step, not out on the flat.
    columns = np.nonzero(moved.any(axis=0))[0]
    assert columns.min() >= 45 and columns.max() <= 55


def test_quantisation_lands_on_the_reporting_step():
    out = DepthSensor(quantisation_mm=1.0).degrade(flat_frame(1.0004), None)
    assert np.allclose(out, 1.0)


def test_the_named_sensors_get_progressively_worse():
    """A sweep needs its reference points ordered, or the curve reads backwards."""
    assert TYPICAL_STEREO.axial_mm_at_1m < HARSH_SPECULAR.axial_mm_at_1m
    assert TYPICAL_STEREO.dropout < HARSH_SPECULAR.dropout
    assert CLEAN.axial_mm_at_1m == 0.0 and CLEAN.dropout == 0.0


# ------------------------------------------------------- the estimator it feeds


def two_faces(bow_mm: float = 0.0, samples: int = 9000,
              rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic corner: two panels meeting on a line that bows by a known amount.

    Built rather than rendered so the answer is exact and the test does not need
    a capture on disk.
    """
    rng = rng or np.random.default_rng(0)
    height = geo.PART_D
    z = rng.uniform(0.0, height, samples)
    bow = (bow_mm / 1000.0) * np.sin(np.pi * z / height)

    # Face A lies in x = bow(z); face B lies in y = bow(z). They meet on the line
    # (bow(z), bow(z), z), which is the seam, and its departure from straight is
    # exactly bow_mm at the middle.
    half = samples // 2
    a = np.column_stack([bow[:half], rng.uniform(0.0, 0.30, half), z[:half]])
    b = np.column_stack([rng.uniform(0.0, 0.30, samples - half), bow[half:], z[half:]])
    seam = np.column_stack([bow, bow, z])
    return np.vstack([a, b]), seam[np.argsort(z)]


def test_the_faces_recover_a_bowed_seam():
    from cell.seam_track import surface_path

    points, seam = two_faces(bow_mm=1.5)
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.0, 0.0, geo.PART_D])
    heights = np.linspace(0.02, geo.PART_D - 0.02, 12)

    path = surface_path(points, start, end, heights, half_width_m=0.4)
    assert path is not None

    expected = (1.5 / 1000.0) * np.sin(np.pi * heights / geo.PART_D)
    assert np.allclose(path[:, 0], expected, atol=2e-4)
    assert np.allclose(path[:, 1], expected, atol=2e-4)


def test_one_surface_is_refused_rather_than_guessed_at():
    from cell.seam_track import surface_path

    rng = np.random.default_rng(4)
    only_one_face = np.column_stack([
        np.zeros(3000), rng.uniform(0, 0.3, 3000), rng.uniform(0, geo.PART_D, 3000)])
    path = surface_path(only_one_face, np.array([0.0, 0.0, 0.0]),
                        np.array([0.0, 0.0, geo.PART_D]),
                        np.linspace(0.02, 0.4, 8), half_width_m=0.4)
    assert path is None


def test_the_inlier_band_stays_under_the_sheet_thickness():
    """A band wider than the sheet lets one plane claim both of a panel's faces.

    This is not a style point. The default used to be 2.5 mm on 2 mm sheet, and
    the resulting offset sat between the two surfaces, biasing the seam by about
    1.4 mm.
    """
    import inspect

    from cell.seam_track import surface_path

    source = inspect.getsource(surface_path)
    assert "0.6 * geo.PART_T" in source
    assert 0.6 * geo.PART_T < geo.PART_T
