"""Welding where the joint is going to be.

The first version of this sent the torch the wrong way. It was caught by the
study rather than by a test, because a sign error shows up as a penalty that does
not scale with the distortion, and nothing here was checking the direction. These
check it, along with the two properties the correction has to have to be safe:
it does nothing when told to do nothing, and it grows with height rather than
being applied as a lump.
"""

from __future__ import annotations

import numpy as np
import pytest

from cell import geometry as geo
from cell.compensate import (
    compensated_path,
    inward_axis,
    where_the_seam_ends_up,
)
from cell.thermal import ThermalDistortion, WeldPass


def straight_seam(stations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    floor = geo.TABLE_TOP + geo.PART_T
    heights = np.linspace(0.02, geo.PART_D - 0.02, stations)
    corner = geo.nominal_joint()
    path = np.column_stack([
        np.full(stations, corner[0]),
        np.full(stations, corner[1]),
        floor + heights,
    ])
    # From the strike, not from the fold. See compensated_path.
    return path, (path[:, 2] - path[:, 2].min()) * 1000.0


def distortion(length_mm: float) -> ThermalDistortion:
    return ThermalDistortion(weld=WeldPass(), thickness_mm=geo.PART_T * 1000.0,
                             seam_length_mm=length_mm)


def test_the_corner_moves_on_a_diagonal_with_no_vertical_component():
    """Both panels rotate about their own fold lines and contribute equally."""
    axis = inward_axis()
    assert axis[2] == 0.0
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert np.isclose(abs(axis[0]), abs(axis[1]))


def test_the_correction_follows_the_joint_rather_than_opposing_it():
    """The bug this file exists for: the torch must go the way the metal goes."""
    path, along = straight_seam()
    model = distortion(float(along[-1]))

    moved, _ = compensated_path(path, along, model)
    target = where_the_seam_ends_up(path, model, seam_floor_m=path[:, 2].min())

    # Both must move off the cold path in the SAME direction.
    correction = moved[-1, :2] - path[-1, :2]
    actual = target[-1, :2] - path[-1, :2]
    assert float(correction @ actual) > 0, "the correction opposes the distortion"


def test_compensating_with_the_right_model_lands_on_the_joint():
    path, along = straight_seam()
    model = distortion(float(along[-1]))

    moved, _ = compensated_path(path, along, model)
    target = where_the_seam_ends_up(path, model, seam_floor_m=path[:, 2].min())

    missed = np.linalg.norm(moved[:, :2] - target[:, :2], axis=1) * 1000.0
    assert missed.max() < 0.01, "an exact model should leave nothing behind"


def test_no_gain_leaves_the_path_exactly_alone():
    """The study's control arm runs this code path, so it has to be a true zero."""
    path, along = straight_seam()
    moved, applied = compensated_path(path, along, distortion(float(along[-1])), gain=0.0)
    assert np.array_equal(moved, path)
    assert applied.worst_mm == 0.0


def test_the_correction_grows_up_the_seam():
    """A bend angle is multiplied by height, so the bottom barely moves."""
    path, along = straight_seam()
    _, applied = compensated_path(path, along, distortion(float(along[-1])))

    shifts = applied.shifts_mm
    assert shifts[0] < shifts[-1]
    assert np.all(np.diff(shifts) >= -1e-9), "the correction should not go backwards"
    assert applied.at_finish_mm == pytest.approx(shifts[-1])


def test_a_hotter_pass_needs_a_bigger_correction():
    path, along = straight_seam()
    cool = ThermalDistortion(weld=WeldPass(current_a=90), thickness_mm=2.0,
                             seam_length_mm=float(along[-1]))
    hot = ThermalDistortion(weld=WeldPass(current_a=160), thickness_mm=2.0,
                            seam_length_mm=float(along[-1]))

    _, cool_applied = compensated_path(path, along, cool)
    _, hot_applied = compensated_path(path, along, hot)
    assert hot_applied.worst_mm > cool_applied.worst_mm


def test_a_path_and_its_heights_have_to_match():
    path, along = straight_seam()
    with pytest.raises(ValueError):
        compensated_path(path, along[:-1], distortion(float(along[-1])))
