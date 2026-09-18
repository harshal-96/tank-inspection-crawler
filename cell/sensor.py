"""What a real depth camera returns, as opposed to what the simulation renders.

Rendered depth is exact. A joint finder scored on exact depth has not been scored
on anything, because the only interesting question about this method is where it
stops working, and exact depth never stops working. So this degrades a rendered
frame the way the sensor would, and the campaign is run again at each level until
it breaks.

Three failure modes decide it, and they are not the same as "add some Gaussian
noise". Ranked by how much they hurt this particular measurement:

  clustered dropout      A stereo or structured-light camera returns nothing from
                         a specular surface angled away from it. Folded sheet
                         steel is specular, so the returns come back with holes in
                         them, and the holes are patches rather than speckle
                         because a whole region shares an orientation. This is the
                         failure mode that puts this method on a shop floor or
                         doesn't, and it is the one a flat dropout rate models
                         worst.

  flying pixels          A pixel straddling a depth discontinuity averages the two
                         surfaces and reports a range belonging to neither. Every
                         depth camera does this. It matters here more than
                         anywhere else, because the discontinuity in question IS
                         the joint: the fold where two panels meet is exactly the
                         edge the method has to locate, so its noise sits directly
                         on the quantity being measured.

  axial noise            Triangulation error grows with the square of range, not
                         linearly, because disparity resolution is fixed in pixels
                         and range is inversely proportional to disparity. Quoting
                         a camera's accuracy as a single millimetre figure without
                         a distance beside it is meaningless.

Quantisation is included because most depth cameras report unsigned millimetres,
but it is small against the other three and is here for completeness rather than
because it changes an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DepthSensor:
    """A degradation model with each term named after the physics behind it.

    Defaults are a clean frame: applying this sensor changes nothing, so the
    zero-noise arm of a sweep goes through exactly the same code path as every
    other arm rather than skipping it.
    """

    # Standard deviation of the range error at one metre, in millimetres. The
    # error at any other range scales with the square of the distance.
    axial_mm_at_1m: float = 0.0

    # Fraction of the frame lost to specular dropout, and how large the patches
    # are. A patch radius of one pixel is speckle; real dropout on sheet metal is
    # a handful of pixels across.
    dropout: float = 0.0
    dropout_patch_px: int = 4

    # Fraction of pixels ALONG a depth discontinuity that report an intermediate
    # range. Only pixels near an edge are eligible, which is the point.
    flying_pixels: float = 0.0
    edge_threshold_m: float = 0.01

    # Reporting step, in millimetres. 0 leaves depth continuous.
    quantisation_mm: float = 0.0

    def degrade(self, depth: np.ndarray,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return the frame this sensor would have produced. Invalid pixels are NaN."""
        rng = rng or np.random.default_rng(0)
        out = np.asarray(depth, dtype=np.float64).copy()
        finite = np.isfinite(out) & (out > 0)

        if self.flying_pixels > 0:
            out = self._fly(out, finite, rng)
        if self.axial_mm_at_1m > 0:
            # sigma(z) = sigma(1 m) * z^2, the triangulation scaling.
            sigma = (self.axial_mm_at_1m / 1000.0) * np.square(np.where(finite, out, 1.0))
            out = np.where(finite, out + rng.normal(0.0, 1.0, out.shape) * sigma, out)
        if self.quantisation_mm > 0:
            step = self.quantisation_mm / 1000.0
            out = np.where(finite, np.round(out / step) * step, out)
        if self.dropout > 0:
            out = np.where(self._holes(out.shape, rng), np.nan, out)
        return out

    # ------------------------------------------------------------------ internals

    def _holes(self, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        """A boolean mask of lost pixels, in patches rather than speckle.

        Seeds are thrown at the frame and grown into squares. The seed rate is
        divided by the area of a patch so that the requested dropout fraction is
        what actually comes out, whatever the patch size.
        """
        radius = max(0, int(self.dropout_patch_px) // 2)
        if radius == 0:
            return rng.random(shape) < self.dropout

        area = (2 * radius + 1) ** 2
        seeds = rng.random(shape) < (self.dropout / area)
        mask = np.zeros(shape, dtype=bool)
        rows, cols = np.nonzero(seeds)
        for r, c in zip(rows, cols, strict=True):
            r0, r1 = max(0, r - radius), min(shape[0], r + radius + 1)
            c0, c1 = max(0, c - radius), min(shape[1], c + radius + 1)
            mask[r0:r1, c0:c1] = True
        return mask

    def _fly(self, depth: np.ndarray, finite: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
        """Pull pixels near a depth step toward the surface behind it.

        A real flying pixel lands somewhere between the two surfaces its footprint
        straddles, so the offset is a random fraction of the local step rather
        than a fixed amount.
        """
        filled = np.where(finite, depth, np.nan)
        dr = np.zeros_like(filled)
        dc = np.zeros_like(filled)
        dr[1:-1, :] = filled[2:, :] - filled[:-2, :]
        dc[:, 1:-1] = filled[:, 2:] - filled[:, :-2]
        step = np.nan_to_num(np.maximum(np.abs(dr), np.abs(dc)))

        edge = (step > self.edge_threshold_m) & finite
        chosen = edge & (rng.random(depth.shape) < self.flying_pixels)
        pull = rng.random(depth.shape) * step
        return np.where(chosen, depth + pull, depth)


# Reference points, so a sweep can be read against something real rather than
# against arbitrary numbers. Both are taken from published datasheet figures for
# widely used stereo depth cameras at the standoff this cell uses, and they are
# here to locate the sweep, not to claim a specific product was modelled.
CLEAN = DepthSensor()

TYPICAL_STEREO = DepthSensor(
    axial_mm_at_1m=2.5,
    dropout=0.10,
    dropout_patch_px=4,
    flying_pixels=0.25,
    quantisation_mm=1.0,
)

HARSH_SPECULAR = DepthSensor(
    axial_mm_at_1m=4.0,
    dropout=0.35,
    dropout_patch_px=6,
    flying_pixels=0.40,
    quantisation_mm=1.0,
)
