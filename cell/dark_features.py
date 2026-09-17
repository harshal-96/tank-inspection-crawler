"""Tell a shadow from a crack, by checking the geometry under it.

A dark line in a photograph is ambiguous. A corner shadow, a scorch mark, a
grinding line and an actual crack all read as a dark band, and no amount of
looking harder at the colour image resolves it. Depth does: a shadow lies flat
across the surface, a crack or a gap steps, and a groove dips and comes back.

So every dark feature found in the image is measured against the depth at the
same pixels, and what comes out is evidence rather than a guess. The inspection
model is then shown both, and cannot call a shadow a defect.

Pure numpy. No Isaac, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# How far the surface has to move across a dark band before it stops being a
# shading effect and starts being geometry. Below this a weld bead's own texture
# and sensor noise dominate.
FLAT_TOLERANCE_MM = 0.4


@dataclass
class DarkFeature:
    """One dark band in the image, and what the depth under it says."""

    kind: str
    pixels: int
    centroid_uv: tuple[float, float]
    width_px: int
    height_px: int
    depth_range_mm: Optional[float]
    step_mm: Optional[float]
    dip_mm: Optional[float]
    missing_depth: bool = False

    @property
    def is_geometry(self) -> bool:
        return self.kind != "shadow"

    def describe(self) -> str:
        if self.kind == "shadow":
            return (
                f"dark band at ({self.centroid_uv[0]:.0f}, {self.centroid_uv[1]:.0f}), "
                f"{self.width_px}x{self.height_px} px, surface flat to within "
                f"{self.depth_range_mm:.2f} mm: a shading effect, not geometry"
            )
        if self.kind == "hole":
            return (
                f"dark band at ({self.centroid_uv[0]:.0f}, {self.centroid_uv[1]:.0f}) "
                "returns no depth at all: the surface is not there"
            )
        if self.kind == "step":
            return (
                f"dark band at ({self.centroid_uv[0]:.0f}, {self.centroid_uv[1]:.0f}) "
                f"steps {self.step_mm:.2f} mm: the two sides are at different depths"
            )
        return (
            f"dark band at ({self.centroid_uv[0]:.0f}, {self.centroid_uv[1]:.0f}) "
            f"dips {self.dip_mm:.2f} mm below its surroundings and returns: a groove"
        )


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components, four-connected, without a scipy dependency."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    rows, cols = mask.shape
    for start_r in range(rows):
        for start_c in range(cols):
            if not mask[start_r, start_c] or labels[start_r, start_c]:
                continue
            current += 1
            stack = [(start_r, start_c)]
            labels[start_r, start_c] = current
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    inside = 0 <= nr < rows and 0 <= nc < cols
                    if inside and mask[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = current
                        stack.append((nr, nc))
    return labels, current


def find_dark(image: np.ndarray, threshold: int = 90, min_pixels: int = 60) -> list[np.ndarray]:
    """Connected dark regions in an RGB image, largest first."""
    if image.ndim != 3:
        raise ValueError("expected an RGB image")
    mask = image.astype(int).sum(axis=2) < threshold
    labels, count = _label(mask)
    regions = [labels == i for i in range(1, count + 1)]
    regions = [r for r in regions if r.sum() >= min_pixels]
    regions.sort(key=lambda r: -r.sum())
    return regions


def measure_feature(
    region: np.ndarray,
    depth: np.ndarray,
    flat_tolerance_mm: float = FLAT_TOLERANCE_MM,
    margin_px: int = 6,
) -> DarkFeature:
    """Classify one dark region by the depth across it.

    Compares the depth inside the band against a collar of surrounding surface.
    Flat with its neighbours means shading. A one-sided offset means a step. A
    dip that recovers means a groove. No returns at all means a hole.
    """
    rows, cols = np.nonzero(region)
    centroid = (float(cols.mean()), float(rows.mean()))
    height_px = int(rows.max() - rows.min() + 1)
    width_px = int(cols.max() - cols.min() + 1)

    inside = depth[region]
    inside = inside[np.isfinite(inside) & (inside > 0)]
    if inside.size < 10:
        return DarkFeature("hole", int(region.sum()), centroid, width_px, height_px,
                           None, None, None, missing_depth=True)

    # A collar of surface either side of the band, at the same rows.
    collar = np.zeros_like(region)
    for shift in (-margin_px, margin_px):
        collar |= np.roll(region, shift, axis=1)
    collar &= ~region
    outside = depth[collar]
    outside = outside[np.isfinite(outside) & (outside > 0)]

    depth_range_mm = float((inside.max() - inside.min()) * 1000.0)
    if outside.size < 10:
        kind = "shadow" if depth_range_mm <= flat_tolerance_mm else "groove"
        return DarkFeature(kind, int(region.sum()), centroid, width_px, height_px,
                           depth_range_mm, None, None)

    inside_median = float(np.median(inside))
    left = depth[np.roll(region, -margin_px, axis=1) & ~region]
    right = depth[np.roll(region, margin_px, axis=1) & ~region]
    left = left[np.isfinite(left) & (left > 0)]
    right = right[np.isfinite(right) & (right > 0)]

    step_mm = None
    if left.size >= 10 and right.size >= 10:
        step_mm = float(abs(np.median(left) - np.median(right)) * 1000.0)
    dip_mm = float((inside_median - np.median(outside)) * 1000.0)

    if step_mm is not None and step_mm > flat_tolerance_mm:
        kind = "step"
    elif abs(dip_mm) > flat_tolerance_mm or depth_range_mm > flat_tolerance_mm * 3:
        kind = "groove"
    else:
        kind = "shadow"

    return DarkFeature(kind, int(region.sum()), centroid, width_px, height_px,
                       depth_range_mm, step_mm, dip_mm)


def inspect_dark_features(
    image: np.ndarray,
    depth: np.ndarray,
    threshold: int = 90,
    min_pixels: int = 60,
    limit: int = 6,
) -> list[DarkFeature]:
    """Every dark band in the image, classified by the geometry beneath it."""
    if image.shape[:2] != depth.shape:
        raise ValueError(
            f"image is {image.shape[:2]} and depth is {depth.shape}; they must match"
        )
    return [measure_feature(r, depth) for r in find_dark(image, threshold, min_pixels)[:limit]]


def evidence(features: list[DarkFeature]) -> dict:
    """A compact summary for the inspection model.

    Deliberately says what the geometry shows and nothing about whether the weld
    is good. Calling that is the model's job.
    """
    return {
        "dark_bands_found": len(features),
        "explained_by_shading": sum(1 for f in features if not f.is_geometry),
        "backed_by_geometry": sum(1 for f in features if f.is_geometry),
        "detail": [f.describe() for f in features],
    }
