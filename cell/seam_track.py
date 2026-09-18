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

    def fitted_path(self, degree: int = 3, robust_passes: int = 2,
                    reject_sigma: float = 2.5) -> np.ndarray:
        """The path a low-order curve puts through the stations, which is the one to weld.

        Treating each station as an independent measurement throws away the
        strongest thing known about the part. The seam is a fold line in
        continuous sheet: springback and camber bow it over its whole length, and
        the metal physically cannot jink between one station and the next 20 mm
        away. So the deviation from straight is smooth and low order, and anything
        in a station-by-station reading that is not smooth is the sensor rather
        than the seam.

        A moving average would also suppress that, but it pays for it at the ends,
        where it has nothing to average against and falls back on the edge value,
        and the ends of a seam are exactly where a bow is largest. A polynomial in
        distance along the seam has no such edge, and it is the right shape: a
        cubic covers a single springback arc and the S that camber adds to it.

        Two other things matter and are cheap. Stations differ in how well they
        were measured, so each is weighted by its own fit residual rather than
        counted equally. And a station that disagrees with the curve by more than
        a few robust deviations is dropped and the curve refitted, because a fit
        that ran away on one slice should not be allowed to bend the path the
        torch follows.
        """
        stations = self.measured
        if len(stations) < degree + 2:
            return self.path()

        along = np.array([s.along_m for s in stations], dtype=float)
        points = np.array([s.position for s in stations], dtype=float)

        # A station's uncertainty scales with its own fit residual. The floor
        # keeps a suspiciously perfect slice from dominating every other one.
        residual = np.array([max(s.residual_mm, 0.05) for s in stations], dtype=float)
        weights = 1.0 / np.square(residual)

        span = max(along.max() - along.min(), 1e-9)
        t = (along - along.min()) / span            # conditioned, so the fit is stable
        keep = np.ones(len(stations), dtype=bool)

        curve = points
        for _ in range(max(1, robust_passes)):
            if keep.sum() < degree + 2:
                break
            coefficients = [
                np.polyfit(t[keep], points[keep, axis], degree, w=weights[keep])
                for axis in range(3)
            ]
            curve = np.column_stack([np.polyval(c, t) for c in coefficients])

            off = np.linalg.norm(points - curve, axis=1)
            scale = np.median(np.abs(off - np.median(off))) * 1.4826
            if scale < 1e-9:
                break
            fresh = off < np.median(off) + reject_sigma * scale
            if fresh.sum() < degree + 2 or np.array_equal(fresh, keep):
                break
            keep = fresh

        return curve

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


def surface_path(
    points_world: np.ndarray,
    seam_start: np.ndarray,
    seam_end: np.ndarray,
    heights: np.ndarray,
    half_width_m: float = 0.045,
    degree: int = 2,
    threshold_m: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Where the seam runs, from both faces at once rather than slice by slice.

    Station-by-station measurement has a weakness that only shows up on a real
    sensor. Each station fits its two lines from the few hundred points inside a
    thin slice, and a short segment constrains a line's ANGLE badly. The angular
    error is then multiplied by the distance out to the vertex, so a slice fit
    that looks reasonable puts the corner millimetres away. Thickening the slice
    helps a little and costs resolution; it does not address the cause.

    The cause is that the faces were treated as new evidence at every station,
    when they are two continuous panels seen in tens of thousands of points. A
    folded panel has one orientation. What varies along the seam is how far out
    the panel sits, because springback and camber bow it, and that is a single
    number per height rather than a whole new plane.

    So each face is fitted once, robustly, from every point near the seam, giving
    a normal that tens of thousands of samples agree on. Only the offset is
    allowed to vary, as a low-order polynomial in height, which is exactly the
    freedom a bow needs and no more. The seam at a height is then where the two
    bowed faces and that height meet: three planes, one point, well conditioned.

    Returns None when the cloud does not hold two distinct faces, which is a
    refusal rather than a guess. The caller keeps its slice measurements in that
    case, because a station it cannot improve is better left as measured.
    """
    from cell import geometry as geo
    from cell.depth_joint import Plane, fit_two_planes

    # The inlier band has to be narrower than the sheet is thick, or a panel's
    # two faces are claimed by one plane and its offset lands between them. On
    # 2 mm sheet a 2.5 mm band, which looks like an ordinary RANSAC tolerance,
    # biased the seam by 1.4 mm and made this estimator worse than the slices it
    # was meant to beat. Derived from the geometry rather than tuned.
    if threshold_m is None:
        threshold_m = 0.6 * geo.PART_T

    axis = np.asarray(seam_end, dtype=float) - np.asarray(seam_start, dtype=float)
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None
    axis = axis / length

    start = np.asarray(seam_start, dtype=float)
    along = (points_world - start) @ axis
    radial = points_world - start - np.outer(along, axis)
    near = (np.linalg.norm(radial, axis=1) < half_width_m)
    near &= (along > -0.02) & (along < length + 0.02)
    gathered = points_world[near]
    if len(gathered) < 200:
        return None

    try:
        face_a, face_b = fit_two_planes(gathered, threshold=threshold_m, rng=rng)
    except (RuntimeError, ValueError):
        return None

    def bowed_offset(face: Plane) -> Optional[np.poly1d]:
        """How far the face sits out, as a function of height along the seam."""
        signed = gathered @ face.normal - face.offset
        inliers = np.abs(signed) < threshold_m
        if inliers.sum() < max(50, (degree + 1) * 20):
            return None
        h = (gathered[inliers] - start) @ axis
        d = gathered[inliers] @ face.normal
        # Conditioned on the seam's own length so the fit does not depend on where
        # the part happens to sit in the cell.
        return np.poly1d(np.polyfit(h / length, d, degree))

    offset_a = bowed_offset(face_a)
    offset_b = bowed_offset(face_b)
    if offset_a is None or offset_b is None:
        return None

    matrix = np.stack([face_a.normal, face_b.normal, axis])
    if abs(np.linalg.det(matrix)) < 1e-6:
        return None

    out = []
    for height in np.asarray(heights, dtype=float):
        t = height / length
        rhs = np.array([offset_a(t), offset_b(t), float(start @ axis) + height])
        out.append(np.linalg.solve(matrix, rhs))
    return np.array(out)


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


# How far off the taught path a part can turn up and still be the right part.
#
# Sized from what the fixture actually delivers, not from what it ought to: over
# fifty arrivals the median part lands 23 mm off the taught path, the worst 51 mm.
# A 45 mm window was tried first and lost four parts outright, all of them the
# ones that arrived furthest out, which is precisely the population the tracking
# exists for. Anything past this is something else in shot.
#
# Opening the window costs nothing in accuracy because it is only used to find
# the part. The second pass closes back down to CAPTURE_RANGE_M and tracks from
# where the first pass landed, so a wide search never loosens the measurement.
LOCATE_RANGE_M = 0.075


def find_and_track(
    points_world: np.ndarray,
    taught_start: np.ndarray,
    taught_end: np.ndarray,
    stations: int = 24,
    half_width_m: float = 0.045,
    slice_thickness_m: float = SLICE_THICKNESS_M,
    locate_range_m: float = LOCATE_RANGE_M,
    capture_range_m: float = CAPTURE_RANGE_M,
    refine: bool = True,
) -> SeamTrack:
    """Locate this part's seam first, then track along it.

    The taught path says where the part was supposed to be. A part that arrived a
    centimetre out is still the right part, and a capture window tight enough to
    keep a tracker honest is far too tight to find it in the first place. So this
    is done in two passes, the way a cell does it: open the window wide enough to
    take in anything the fixture can deliver and ask only where this part's corner
    is, then close the window back down and track from there.

    Without the second pass the window has to be left open, and an open window is
    how a tracker ends up following a rim, an edge of the fixture, or the part next
    to it.
    """
    # The first pass has to reach the part wherever it arrived, so its window is
    # the tracking window PLUS however far out the part can be. Widening only the
    # capture range is not enough and was the subtler half of this bug: capture
    # range decides whether a fit is believed, while half width decides which
    # points are gathered in the first place. A part 51 mm off the taught line
    # with a 45 mm gathering window puts just one of its two faces inside the
    # window, and one face is not a corner, so every station is refused with the
    # part sitting in plain view.
    coarse = track_seam(
        points_world,
        taught_start,
        taught_end,
        stations=stations,
        half_width_m=half_width_m + locate_range_m,
        slice_thickness_m=slice_thickness_m,
        capture_range_m=locate_range_m,
    )
    if len(coarse.measured) < 3:
        return coarse

    # Straight line through what the first pass found: this part's corner, shifted
    # and turned as the part arrived. Straight on purpose, because it is only a
    # place to look from. The bend in the seam is what the second pass measures.
    along = np.array([station.along_m for station in coarse.measured])
    found = np.array([station.position for station in coarse.measured])
    design = np.vstack([along, np.ones_like(along)]).T
    fit, *_ = np.linalg.lstsq(design, found, rcond=None)

    start = np.asarray(taught_start, dtype=float)
    end = np.asarray(taught_end, dtype=float)
    length = float(np.linalg.norm(end - start))
    located_start = fit[1]
    located_end = fit[0] * length + fit[1]

    track = track_seam(
        points_world,
        located_start,
        located_end,
        stations=stations,
        half_width_m=half_width_m,
        slice_thickness_m=slice_thickness_m,
        capture_range_m=capture_range_m,
    )
    if refine:
        track = refine_positions(track, points_world, located_start, located_end,
                                 half_width_m=half_width_m)
    return track


def refine_positions(
    track: SeamTrack,
    points_world: np.ndarray,
    seam_start: np.ndarray,
    seam_end: np.ndarray,
    half_width_m: float = 0.045,
) -> SeamTrack:
    """Replace each station's position with the one both whole faces agree on.

    Only the position changes. The gap, the point counts and the residual stay as
    the slice measured them, because those are local by nature: a gap is how far
    short the metal stops HERE, and nothing about the rest of the seam can tell
    you that. Where the seam runs is the opposite kind of quantity, and is better
    estimated from every point on both panels than from the few hundred inside one
    slice.

    Silently does nothing when the faces cannot be fitted. A station that cannot
    be improved is left as measured rather than dropped.
    """
    measured = track.measured
    if len(measured) < 4:
        return track

    heights = np.array([s.along_m for s in measured])
    refined = surface_path(points_world, seam_start, seam_end, heights,
                           half_width_m=half_width_m)
    if refined is None:
        return track

    # A refinement that disagrees with the slices by more than the capture range
    # is not a refinement, it is a different seam. Keep the slices in that case.
    moved = np.linalg.norm(refined - np.array([s.position for s in measured]), axis=1)
    if float(np.median(moved)) > CAPTURE_RANGE_M:
        return track

    for station, position in zip(measured, refined, strict=True):
        station.position = position
    return track
