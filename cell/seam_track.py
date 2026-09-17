"""Track the seam along its length, ahead of the arc.

The part is a flat blank, cut and then folded up. At each corner the two folded
edges butt against each other and the robot welds that seam. Three things move it
away from the taught path: springback leaves the bend angle slightly open, camber
bows the blank, and heat from an earlier weld pulls the edge. A taught program
does not know about any of them, so the torch wanders off the seam and gaps form.

The seam therefore is NOT a straight line, and must not be measured as one.
Fitting a plane to each face and intersecting them gives a straight answer by
construction: it averages away the very deviation that causes the problem. So the
seam is walked instead, and measured at every station along it.

Each station yields two things:

    where the seam actually is    the point the torch should pass through
    how wide the gap is           whether the fold is good enough to weld at all

The gap is the press check, measured rather than eyeballed. Zero means the edges
are touching. A gap opening over part of the length means the brake is off, and
that is a decision before the arc strikes, because a wide gap is what gives you
burn-through.

Because the weld itself pulls the edge, measurement leads the torch: `ahead_of`
returns the seam where the torch is about to be, not where it has been.

Pure numpy. No Isaac, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# A slice thinner than this holds too few points to fit; thicker and it smears
# real curvature of the seam back into a straight line.
SLICE_THICKNESS_M = 0.004

# Material is expected right up to the vertex. Beyond this, the edges are not
# touching and there is a gap.
TOUCHING_TOLERANCE_MM = 0.3

# How far from the taught path the seam is allowed to be found. Springback, camber
# and weld distortion move a seam by millimetres, not centimetres, so a result
# further out than this is a bad fit rather than a real seam. Every seam tracker
# has a capture range for the same reason: at the ends of a seam the slice runs
# short of material and the fit can run away without anything else noticing.
CAPTURE_RANGE_M = 0.015


@dataclass
class SeamStation:
    """The seam at one point along its length."""

    along_m: float
    position: np.ndarray
    gap_mm: float
    face_a_points: int
    face_b_points: int
    residual_mm: float
    valid: bool = True
    reason: Optional[str] = None

    @property
    def touching(self) -> bool:
        return self.valid and self.gap_mm <= TOUCHING_TOLERANCE_MM


@dataclass
class SeamTrack:
    stations: list[SeamStation] = field(default_factory=list)

    @property
    def measured(self) -> list[SeamStation]:
        return [s for s in self.stations if s.valid]

    def path(self) -> np.ndarray:
        """The points the torch should follow, in order along the seam."""
        return np.array([s.position for s in self.measured])

    def gaps_mm(self) -> np.ndarray:
        return np.array([s.gap_mm for s in self.measured])

    def ahead_of(self, along_m: float, lead_m: float) -> Optional[SeamStation]:
        """The seam at a set distance in front of the torch.

        The weld pulls the edge as it goes, so the useful measurement is of metal
        that has not been welded yet.
        """
        candidates = [s for s in self.measured if s.along_m >= along_m + lead_m]
        return candidates[0] if candidates else None

    def smoothed_path(self, window: int = 5) -> np.ndarray:
        """The path with sensor noise taken out, which is what the torch follows.

        A station-by-station measurement carries the sensor's noise, and a torch
        driven straight off it would chatter along the seam. A short moving
        average keeps the real bow, which spans the whole seam, and drops the
        jitter, which does not.
        """
        points = self.path()
        if len(points) < 3:
            return points
        window = max(3, min(window | 1, len(points)))
        half = window // 2
        padded = np.pad(points, ((half, half), (0, 0)), mode="edge")
        kernel = np.ones(window) / window
        return np.column_stack(
            [np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)]
        )

    def wander_mm(self, smooth: bool = True) -> float:
        """How far the seam departs from the straight line between its own ends.

        This is the number a taught program gets wrong, because a taught program
        is that straight line.
        """
        points = self.smoothed_path() if smooth else self.path()
        if len(points) < 3:
            return 0.0
        start, end = points[0], points[-1]
        axis = end - start
        length = np.linalg.norm(axis)
        if length < 1e-9:
            return 0.0
        axis = axis / length
        offsets = points - start
        along = offsets @ axis
        perpendicular = offsets - np.outer(along, axis)
        return float(np.linalg.norm(perpendicular, axis=1).max() * 1000.0)

    def summary(self) -> dict:
        """What the cell and the inspector need, without the raw points."""
        measured = self.measured
        if not measured:
            return {"stations_measured": 0, "of": len(self.stations)}
        gaps = self.gaps_mm()
        worst = int(np.argmax(gaps))
        return {
            "stations_measured": len(measured),
            "of": len(self.stations),
            "seam_length_mm": round((measured[-1].along_m - measured[0].along_m) * 1000.0, 1),
            "gap_mm_mean": round(float(gaps.mean()), 3),
            "gap_mm_max": round(float(gaps.max()), 3),
            "gap_worst_at_mm": round(measured[worst].along_m * 1000.0, 1),
            "stations_touching": sum(1 for s in measured if s.touching),
            "path_wander_mm": round(self.wander_mm(), 3),
        }


def _fit_line(points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares line through 2D points: a point on it, a unit direction, rms."""
    centroid = points_2d.mean(axis=0)
    centred = points_2d - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    direction = vh[0] / np.linalg.norm(vh[0])
    normal = np.array([-direction[1], direction[0]])
    residual = float(np.sqrt(np.mean((centred @ normal) ** 2)))
    return centroid, direction, residual


def _fit_line_ransac(
    points_2d: np.ndarray,
    threshold: float = 0.0008,
    iterations: int = 120,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dominant line and its inliers, robust to the other arm of the L."""
    rng = rng or np.random.default_rng(0)
    best_inliers = np.zeros(len(points_2d), dtype=bool)
    for _ in range(iterations):
        i, j = rng.choice(len(points_2d), size=2, replace=False)
        span = points_2d[j] - points_2d[i]
        norm = np.linalg.norm(span)
        if norm < 1e-9:
            continue
        direction = span / norm
        normal = np.array([-direction[1], direction[0]])
        inliers = np.abs((points_2d - points_2d[i]) @ normal) < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    return points_2d[best_inliers], best_inliers


def _split_by_angle(points_2d: np.ndarray, min_angle_deg: float = 25.0):
    """Separate an L of points into its two arms.

    Fitting the dominant line least-squares does not work here: with both faces
    equally represented the best single line runs along the diagonal of the L and
    claims neither arm. One arm has to be taken robustly, then the other fitted to
    what is left.
    """
    a, inliers = _fit_line_ransac(points_2d)
    leftover = points_2d[~inliers]
    if len(a) < 4 or len(leftover) < 4:
        return None

    # The leftover is not the second edge: it is the second edge plus whatever the
    # first fit threw out, and fitting that mixture least-squares gives a residual
    # several times the real noise. That residual sets the tolerance used to
    # decide which edge a point belongs to, so an inflated one lets each edge
    # claim the other's material right up to the vertex and the gap reads zero on
    # a seam that is plainly open. The second edge has to be fitted robustly too.
    b, _ = _fit_line_ransac(leftover)
    if len(b) < 4:
        return None

    _, dir_a, res_a = _fit_line(a)
    _, dir_b, res_b = _fit_line(b)
    cosine = abs(float(dir_a @ dir_b))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if angle < min_angle_deg:
        return None
    return a, b, res_a, res_b


def _vertex(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    point_a, dir_a, _ = _fit_line(a)
    point_b, dir_b, _ = _fit_line(b)
    matrix = np.column_stack([dir_a, -dir_b])
    if abs(np.linalg.det(matrix)) < 1e-9:
        return None
    t = np.linalg.solve(matrix, point_b - point_a)
    return point_a + t[0] * dir_a


def _edge_reach(distances: np.ndarray, bin_m: float = 0.00025) -> float:
    """Where the material actually starts, as a distance from the vertex.

    Not the smallest distance. The smallest of a thousand noisy samples sits
    several standard deviations below the true edge, which on a sub-millimetre
    sensor understates each edge by close to a millimetre and halves the reported
    gap. Not a quantile either: the points run all the way out along the face, so
    a quantile of them measures the face, not its edge.

    Density does it without needing to know the noise. Bin the distances and take
    the first bin that holds a real share of the material's typical density. A few
    stray samples thrown inward by noise cannot fill a bin; the edge can.
    """
    if distances.size == 0:
        return 0.0
    span = float(distances.max())
    if span < bin_m:
        return 0.0
    counts, edges = np.histogram(distances, bins=np.arange(0.0, span + bin_m, bin_m))
    occupied = counts[counts > 0]
    if occupied.size == 0:
        return 0.0
    threshold = 0.5 * float(np.median(occupied))
    real = np.nonzero(counts >= threshold)[0]
    if real.size == 0:
        return float(distances.min())
    # The centre of the first occupied bin, not its near edge. Taking the near
    # edge biases every reading half a bin short, and on a gap measurement short
    # is the dangerous direction: it reports edges as touching when they are not,
    # which passes a part the press got wrong.
    return float(edges[real[0]] + bin_m / 2.0)


def measure_station(
    slice_points: np.ndarray,
    lateral: np.ndarray,
    depth_axis: np.ndarray,
    origin: np.ndarray,
) -> tuple[Optional[np.ndarray], float, int, int, float, Optional[str]]:
    """One cross-section: find where the two edges meet and how far apart they are.

    Points are flattened onto the plane of the cut, the two folded faces are
    separated, and their lines intersected. The gap is how far short of that
    vertex each face's material actually stops: metal running right up to it means
    the edges are touching.
    """
    if len(slice_points) < 12:
        return None, 0.0, 0, 0, 0.0, "too few points in this slice"

    offsets = slice_points - origin
    flat = np.column_stack([offsets @ lateral, offsets @ depth_axis])

    split = _split_by_angle(flat)
    if split is None:
        return None, 0.0, 0, 0, 0.0, "only one face visible in this slice"
    a, b, res_a, res_b = split

    vertex = _vertex(a, b)
    if vertex is None:
        return None, 0.0, len(a), len(b), 0.0, "the two faces are parallel here"

    # Re-assign every point by its distance to each fitted line, rather than by
    # which arm the robust fit happened to claim it for. A sample sitting on the
    # vertex lies on BOTH lines and belongs to both edges; counting it for only
    # one makes the other look as though it stops a sample short, and reports a
    # gap on a seam whose edges are touching.
    _, dir_a, _ = _fit_line(a)
    _, dir_b, _ = _fit_line(b)
    normal_a = np.array([-dir_a[1], dir_a[0]])
    normal_b = np.array([-dir_b[1], dir_b[0]])
    tolerance = max(2.0 * max(res_a, res_b), 0.0006)
    on_a = np.abs((flat - vertex) @ normal_a) < tolerance
    on_b = np.abs((flat - vertex) @ normal_b) < tolerance
    if on_a.sum() < 3 or on_b.sum() < 3:
        return None, 0.0, len(a), len(b), 0.0, "one edge has too little material here"

    # How close each edge's material comes to the vertex. Material running right
    # up to it means the edges are touching.
    # Measured ALONG each edge, not as a straight-line distance from the vertex.
    # Range noise pushes points across a face, not along it, so a distance that
    # mixes the two carries the noise into the answer. The along-edge coordinate
    # does not move when the depth reading wobbles.
    reach_a = _edge_reach(np.abs((flat[on_a] - vertex) @ dir_a))
    reach_b = _edge_reach(np.abs((flat[on_b] - vertex) @ dir_b))
    gap_mm = (reach_a + reach_b) * 1000.0

    position = origin + vertex[0] * lateral + vertex[1] * depth_axis
    residual_mm = max(res_a, res_b) * 1000.0
    return position, gap_mm, len(a), len(b), residual_mm, None


def track_seam(
    points_world: np.ndarray,
    seam_start: np.ndarray,
    seam_end: np.ndarray,
    stations: int = 24,
    half_width_m: float = 0.045,
    slice_thickness_m: float = SLICE_THICKNESS_M,
    capture_range_m: float = CAPTURE_RANGE_M,
) -> SeamTrack:
    """Walk the taught seam and measure the real one at every station.

    seam_start and seam_end are the taught path, used only to say where to look
    and which way the seam runs. What comes back is measured, and is free to
    depart from that line: that departure is the whole point.
    """
    axis = np.asarray(seam_end, dtype=float) - np.asarray(seam_start, dtype=float)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        raise ValueError("seam_start and seam_end are the same point")
    axis = axis / length

    # A frame across the seam: one axis lateral, one along the line of sight.
    reference = np.array([0.0, 0.0, 1.0])
    if abs(axis @ reference) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])
    lateral = np.cross(axis, reference)
    lateral /= np.linalg.norm(lateral)
    depth_axis = np.cross(axis, lateral)
    depth_axis /= np.linalg.norm(depth_axis)

    along_all = (points_world - seam_start) @ axis
    radial = points_world - seam_start - np.outer(along_all, axis)
    within = np.linalg.norm(radial, axis=1) < half_width_m

    track = SeamTrack()
    for index in range(stations):
        along = length * (index + 0.5) / stations
        band = within & (np.abs(along_all - along) < slice_thickness_m / 2.0)
        origin = seam_start + along * axis

        position, gap, count_a, count_b, residual, reason = measure_station(
            points_world[band], lateral, depth_axis, origin
        )
        if position is not None:
            strayed = float(np.linalg.norm(position - origin))
            if strayed > capture_range_m:
                position, reason = None, (
                    f"fit landed {strayed * 1000:.0f} mm from the taught path, "
                    "outside the capture range"
                )
        track.stations.append(
            SeamStation(
                along_m=along,
                position=position if position is not None else origin,
                gap_mm=gap,
                face_a_points=count_a,
                face_b_points=count_b,
                residual_mm=residual,
                valid=position is not None,
                reason=reason,
            )
        )
    return track
