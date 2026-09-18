"""Weld where the joint is going to be, not where it was.

The measurement is cold. The weld is not. Heat put into the bottom of a seam
rotates both folded sides about their fold lines, so the top of the joint walks
inward long before the torch reaches it: on this part, 1.84 mm by the time the arc
finishes. A cell that measures a point and then drives to it is aiming at where
that point used to be, which is a thing the thermal model in this repository has
said in a docstring for weeks while nothing acted on it.

So the path is offset by the distortion predicted for the moment the arc arrives
at each station. Two things make that honest rather than circular:

  the prediction is not the truth    The model has calibrated constants and they
                                     carry error. Compensating with a model that
                                     is exactly right is a tautology, so the
                                     study that scores this drives the cell with
                                     nominal constants while the metal moves by
                                     perturbed ones, and reports how wrong the
                                     model can be before compensating stops
                                     helping.

  over-correcting is a real failure  Pushing the torch past the joint is as bad as
                                     missing it behind. The benefit curve has two
                                     sides and both are reported.

Direction. Both folded sides rotate about their own fold lines, the +x panel
carrying its top edge toward -x and the -y panel carrying its toward +y, so the
corner they share moves along the inward diagonal with equal contributions from
each. That is the same axis the drift diagnosis projects onto, and it is derived
here from the part's own geometry rather than written down twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cell.thermal import ThermalDistortion


def inward_axis() -> np.ndarray:
    """Unit vector, in world xy, along which the welded corner is pulled.

    Equal and opposite contributions from the two panels, which is what makes it
    a diagonal regardless of how much longer the tray is than it is wide.
    """
    axis = np.array([-1.0, 1.0, 0.0])
    return axis / math.sqrt(2.0)


@dataclass(frozen=True)
class Compensation:
    """What was added to the path, so it can be inspected rather than assumed."""

    shifts_mm: np.ndarray
    worst_mm: float
    at_finish_mm: float

    def describe(self) -> str:
        return (f"path pushed out by up to {self.worst_mm:.2f} mm to meet the joint "
                f"where the arc will have left it, {self.at_finish_mm:.2f} mm at the top")


def compensated_path(
    path: np.ndarray,
    along_mm: np.ndarray,
    distortion: ThermalDistortion,
    gain: float = 1.0,
) -> tuple[np.ndarray, Compensation]:
    """Offset each station by the distortion expected when the torch gets there.

    along_mm is distance from where the WELD starts, not from the fold at the
    bottom of the panel. The thermal model counts heat from the strike, so a path
    whose first station sits 20 mm up the panel starts at 0 here, not at 20. The
    two conventions differ by exactly the standoff of the first station, which is
    small enough to look like a modelling residual and is not one: a test that
    compensated with an exact model and still found 0.17 mm left over is what
    turned it up.

    The arc is at a station's own height when it welds it, so the shift applied at
    a station is the one the model predicts for that moment, which already
    includes the heat put in below it.

    gain scales the correction. One applies the model as it stands; zero is no
    compensation and runs the same code path, which is what the study's control
    arm uses so the two arms differ only in this number.
    """
    path = np.asarray(path, dtype=float)
    along_mm = np.asarray(along_mm, dtype=float)
    if len(path) != len(along_mm):
        raise ValueError("a path and its station heights must be the same length")

    shifts = np.array([
        distortion.seam_shift_mm(float(height), arc_mm=float(height))
        for height in along_mm
    ]) * gain

    # The torch follows the joint. The correction goes the same way the metal
    # goes, not against it: the point of this is to arrive where the joint will
    # be, and sending the torch the other way doubles the miss rather than
    # cancelling it. Written down because the first version had it backwards and
    # the study caught it as a constant penalty that did not scale with the
    # distortion, which is what a sign error looks like and a modelling error
    # does not.
    offset = np.outer(shifts / 1000.0, inward_axis())
    moved = path + offset

    return moved, Compensation(
        shifts_mm=shifts,
        worst_mm=float(np.max(np.abs(shifts))) if len(shifts) else 0.0,
        at_finish_mm=float(shifts[-1]) if len(shifts) else 0.0,
    )


def where_the_seam_ends_up(
    truth: np.ndarray,
    distortion: ThermalDistortion,
    seam_floor_m: Optional[float] = None,
) -> np.ndarray:
    """The cold seam, moved by the heat, which is what the torch actually meets.

    Used by the study to build a target. Takes a distortion whose constants may
    differ from the ones the cell compensated with, because a model that is right
    by construction proves nothing.

    Heights are measured from seam_floor_m, which must be where the weld starts,
    matching the convention compensated_path uses. Passing anything else silently
    shifts one of the two curves against the other.
    """
    truth = np.asarray(truth, dtype=float)
    if len(truth) == 0:
        return truth
    floor = truth[:, 2].min() if seam_floor_m is None else seam_floor_m
    heights_mm = (truth[:, 2] - floor) * 1000.0

    shifts = np.array([
        distortion.seam_shift_mm(float(h), arc_mm=float(h)) for h in heights_mm
    ])
    return truth + np.outer(shifts / 1000.0, inward_axis())
