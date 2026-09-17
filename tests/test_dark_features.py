"""Tests for telling a shadow from a crack.

Scenes are synthesised so the answer is known: the same dark band in the image,
with different geometry underneath it, must come out classified differently. That
is the whole claim.
"""

from __future__ import annotations

import numpy as np
import pytest

from cell.dark_features import (
    DarkFeature,
    evidence,
    find_dark,
    inspect_dark_features,
    measure_feature,
)

SIZE = 80
BAND = slice(36, 44)  # a dark band eight pixels wide, down the middle


def scene(depth_fn) -> tuple[np.ndarray, np.ndarray]:
    """A pale surface with a dark vertical band, and whatever depth is asked for."""
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    image[:, BAND] = 10

    depth = np.full((SIZE, SIZE), 1.0)
    for col in range(SIZE):
        depth[:, col] = depth_fn(col)
    return image, depth


def only_feature(image, depth) -> DarkFeature:
    features = inspect_dark_features(image, depth, min_pixels=20)
    assert len(features) == 1
    return features[0]


# --------------------------------------------------------------------- detection


def test_finds_the_dark_band():
    image, depth = scene(lambda c: 1.0)
    regions = find_dark(image, min_pixels=20)
    assert len(regions) == 1
    assert regions[0].sum() == SIZE * (BAND.stop - BAND.start)


def test_ignores_specks_below_the_size_floor():
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    image[10:12, 10:12] = 0  # four pixels
    assert find_dark(image, min_pixels=20) == []


def test_separates_two_bands():
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    image[:, 10:16] = 0
    image[:, 60:66] = 0
    assert len(find_dark(image, min_pixels=20)) == 2


def test_rejects_mismatched_shapes():
    image, _ = scene(lambda c: 1.0)
    with pytest.raises(ValueError, match="must match"):
        inspect_dark_features(image, np.ones((10, 10)))


# ----------------------------------------------------------------- the real claim


def test_flat_depth_is_a_shadow():
    """The band is as dark as any crack, and the surface under it is flat."""
    image, depth = scene(lambda c: 1.0)
    feature = only_feature(image, depth)
    assert feature.kind == "shadow"
    assert feature.is_geometry is False
    assert feature.depth_range_mm == pytest.approx(0.0, abs=1e-6)


def test_a_step_in_depth_is_geometry():
    """Identical image, but the two sides sit at different depths."""
    image, depth = scene(lambda c: 1.0 if c < 40 else 1.004)
    feature = only_feature(image, depth)
    assert feature.kind == "step"
    assert feature.is_geometry is True
    assert feature.step_mm == pytest.approx(4.0, abs=0.5)


def test_a_dip_that_recovers_is_a_groove():
    image, depth = scene(lambda c: 1.0015 if BAND.start <= c < BAND.stop else 1.0)
    feature = only_feature(image, depth)
    assert feature.kind == "groove"
    assert feature.is_geometry is True
    assert feature.dip_mm == pytest.approx(1.5, abs=0.3)


def test_no_depth_at_all_is_a_hole():
    image, depth = scene(lambda c: 1.0)
    depth[:, BAND] = np.nan
    feature = only_feature(image, depth)
    assert feature.kind == "hole"
    assert feature.missing_depth is True
    assert feature.is_geometry is True


def test_the_same_image_gives_different_answers_on_different_geometry():
    """The point of the whole module: colour alone cannot decide this."""
    flat_image, flat_depth = scene(lambda c: 1.0)
    step_image, step_depth = scene(lambda c: 1.0 if c < 40 else 1.004)

    assert np.array_equal(flat_image, step_image)
    assert only_feature(flat_image, flat_depth).kind == "shadow"
    assert only_feature(step_image, step_depth).kind == "step"


def test_noise_below_tolerance_stays_a_shadow():
    rng = np.random.default_rng(0)
    image, depth = scene(lambda c: 1.0)
    depth = depth + rng.normal(0, 0.00005, depth.shape)  # 0.05 mm of jitter
    assert only_feature(image, depth).kind == "shadow"


# -------------------------------------------------------------------- reporting


def test_evidence_counts_both_kinds():
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    image[:, 10:18] = 0
    image[:, 55:63] = 0
    depth = np.full((SIZE, SIZE), 1.0)
    depth[:, 55:] = 1.004  # only the second band is real geometry

    summary = evidence(inspect_dark_features(image, depth, min_pixels=20))
    assert summary["dark_bands_found"] == 2
    assert summary["explained_by_shading"] == 1
    assert summary["backed_by_geometry"] == 1


def test_descriptions_say_what_was_measured():
    image, depth = scene(lambda c: 1.0)
    text = only_feature(image, depth).describe()
    assert "flat" in text and "mm" in text


def test_evidence_of_nothing_is_empty_not_an_error():
    assert evidence([])["dark_bands_found"] == 0


def test_measure_feature_handles_a_region_touching_the_edge():
    image = np.full((SIZE, SIZE, 3), 200, dtype=np.uint8)
    image[:, :6] = 0
    depth = np.full((SIZE, SIZE), 1.0)
    feature = measure_feature(find_dark(image, min_pixels=20)[0], depth)
    assert feature.kind in {"shadow", "groove", "step"}
