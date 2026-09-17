"""Find the weld joint from depth, by geometry rather than by training.

A fillet joint is the line where two panel faces meet. Given a depth image, the
two faces are planes in the point cloud, and the joint is their intersection. That
is metric by construction: no pixel-to-millimetre calibration to fit, no labelled
images, and nothing to retrain when the part changes, because every fillet joint
is still two planes meeting.

This is the principle a laser seam tracker works on. The interesting question is
not whether it works on perfect data, which it does trivially, but how it degrades
as sensor noise rises. `noisy_depth` exists so that degradation can be measured
rather than assumed.

Pure numpy. No Isaac, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera(
        cls,
        width: int,
        height: int,
        focal_length_mm: float,
        horizontal_aperture_mm: float,
        vertical_aperture_mm: Optional[float] = None,
    ) -> "Intrinsics":
        """Build intrinsics from the physical camera parameters USD stores."""
        vertical_aperture_mm = vertical_aperture_mm or horizontal_aperture_mm
        return cls(
            width=width,
            height=height,
            fx=focal_length_mm / horizontal_aperture_mm * width,
            fy=focal_length_mm / vertical_aperture_mm * height,
            cx=width / 2.0,
            cy=height / 2.0,
        )


@dataclass(frozen=True)
class Plane:
    """The set of points p where normal . p == offset. Normal is unit length."""

    normal: np.ndarray
    offset: float

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Signed perpendicular distance from each point to the plane."""
        return points @ self.normal - self.offset


def backproject(
    depth: np.ndarray, intrinsics: Intrinsics, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Depth image to an (N, 3) point cloud in the camera frame.

    Depth is distance along the optical axis, which is what a depth camera and
    Isaac's distance-to-image-plane annotator both report. The camera looks down
    its own -z, with +x right and +y up, so image rows run opposite to +y.
    """
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(
            f"depth is {depth.shape}, intrinsics describe "
            f"{(intrinsics.height, intrinsics.width)}"
        )

    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask.astype(bool)

    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    x = (cols - intrinsics.cx) * z / intrinsics.fx
    y = -(rows - intrinsics.cy) * z / intrinsics.fy
    return np.stack([x, y, -z], axis=1)


def transform_points(points: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) cloud."""
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")
    homogeneous = np.hstack([points, np.ones((len(points), 1))])
    return (homogeneous @ camera_to_world.T)[:, :3]


def fit_plane(points: np.ndarray) -> Plane:
    """Least-squares plane through the points, via the smallest principal axis."""
    if len(points) < 3:
        raise ValueError("need at least three points to fit a plane")
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    return Plane(normal=normal, offset=float(normal @ centroid))


def fit_plane_ransac(
    points: np.ndarray,
    threshold: float = 0.002,
    iterations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Plane, np.ndarray]:
    """Dominant plane plus its inlier mask, robust to the other panel and to noise."""
    if len(points) < 3:
        raise ValueError("need at least three points to fit a plane")
    rng = rng or np.random.default_rng(0)

    best_plane, best_inliers = None, np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        sample = points[rng.choice(len(points), size=3, replace=False)]
        edge_a, edge_b = sample[1] - sample[0], sample[2] - sample[0]
        normal = np.cross(edge_a, edge_b)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:  # collinear sample
            continue
        normal = normal / norm
        candidate = Plane(normal=normal, offset=float(normal @ sample[0]))

        inliers = np.abs(candidate.distance(points)) < threshold
        if inliers.sum() > best_inliers.sum():
            best_plane, best_inliers = candidate, inliers

    if best_plane is None or best_inliers.sum() < 3:
        raise RuntimeError("no plane found; check the threshold against the noise level")

    # Refit on every inlier: the three-point sample only selects the support.
    return fit_plane(points[best_inliers]), best_inliers


def fit_two_planes(
    points: np.ndarray,
    threshold: float = 0.002,
    iterations: int = 200,
    min_angle_deg: float = 25.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Plane, Plane]:
    """The two panel faces. The second is fitted to whatever the first did not claim.

    Rejects a second plane that is nearly parallel to the first, which means the
    cloud held one surface rather than a joint.
    """
    rng = rng or np.random.default_rng(0)
    first, inliers = fit_plane_ransac(points, threshold, iterations, rng)

    remaining = points[~inliers]
    if len(remaining) < 3:
        raise RuntimeError("only one surface in view; no joint to find")
    second, _ = fit_plane_ransac(remaining, threshold, iterations, rng)

    cosine = abs(float(first.normal @ second.normal))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    if angle < min_angle_deg:
        raise RuntimeError(
            f"the two surfaces differ by only {angle:.1f} degrees, "
            "which is not a joint"
        )
    return first, second


def intersect_planes(a: Plane, b: Plane) -> tuple[np.ndarray, np.ndarray]:
    """The line where two planes meet, as a point on it and a unit direction."""
    direction = np.cross(a.normal, b.normal)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        raise ValueError("planes are parallel and do not intersect in a line")
    direction = direction / norm

    # One point on the line: the one that also lies on the plane through the
    # origin perpendicular to the line, which makes the system well conditioned.
    matrix = np.stack([a.normal, b.normal, direction])
    point = np.linalg.solve(matrix, np.array([a.offset, b.offset, 0.0]))
    return point, direction


def joint_at_height(point: np.ndarray, direction: np.ndarray, z: float) -> tuple[float, float]:
    """Where the joint line crosses a given height, as (x, y)."""
    if abs(direction[2]) < 1e-9:
        raise ValueError("the joint line is horizontal; it has no single crossing at z")
    t = (z - point[2]) / direction[2]
    crossing = point + t * direction
    return (float(crossing[0]), float(crossing[1]))


def find_joint(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    camera_to_world: np.ndarray,
    at_height: float,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.002,
    iterations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """Depth image in, joint position at a given height out, in world metres."""
    cloud = backproject(depth, intrinsics, mask)
    if len(cloud) < 6:
        raise RuntimeError("not enough valid depth samples to find a joint")
    world = transform_points(cloud, camera_to_world)
    first, second = fit_two_planes(world, threshold, iterations, rng=rng)
    point, direction = intersect_planes(first, second)
    return joint_at_height(point, direction, at_height)


def noisy_depth(
    depth: np.ndarray,
    sigma_m: float,
    rng: Optional[np.random.Generator] = None,
    dropout: float = 0.0,
) -> np.ndarray:
    """Add Gaussian range noise and random dropout, as a real sensor would.

    Simulated depth is exact, so a joint finder scored on it measures nothing.
    Accuracy is only meaningful as a curve against the noise a real sensor has.
    """
    rng = rng or np.random.default_rng(0)
    out = depth + rng.normal(0.0, sigma_m, size=depth.shape) if sigma_m > 0 else depth.copy()
    if dropout > 0:
        out = np.where(rng.random(depth.shape) < dropout, np.nan, out)
    return out


@dataclass
class JointMeasurement:
    """One look at the joint: the path to weld, and how much to trust it.

    The quality signals exist so that something else can decide whether this look
    was good enough or whether the part should be presented differently. They are
    all derived from the fit, not from ground truth, so they are available on a
    real cell exactly as they are here.
    """

    waypoints: np.ndarray
    point: np.ndarray
    direction: np.ndarray
    face_a_inliers: int
    face_b_inliers: int
    face_a_rms_mm: float
    face_b_rms_mm: float
    included_angle_deg: float
    path_length_mm: float

    def at_height(self, z: float) -> tuple[float, float]:
        return joint_at_height(self.point, self.direction, z)

    @property
    def inlier_balance(self) -> float:
        """0 when one face is invisible, 1 when both are seen equally.

        A face viewed almost edge-on contributes few points and fits badly, which
        is the failure that drags a joint estimate off.
        """
        total = self.face_a_inliers + self.face_b_inliers
        if total == 0:
            return 0.0
        return 2.0 * min(self.face_a_inliers, self.face_b_inliers) / total

    def summary(self) -> dict:
        """Compact, JSON-safe, and the only thing the planner is shown."""
        return {
            "included_angle_deg": round(self.included_angle_deg, 1),
            "face_a_points": self.face_a_inliers,
            "face_b_points": self.face_b_inliers,
            "inlier_balance": round(self.inlier_balance, 3),
            "face_a_fit_rms_mm": round(self.face_a_rms_mm, 3),
            "face_b_fit_rms_mm": round(self.face_b_rms_mm, 3),
            "path_length_mm": round(self.path_length_mm, 1),
            "waypoints": len(self.waypoints),
        }


def _rms_mm(plane: Plane, points: np.ndarray) -> float:
    return float(np.sqrt(np.mean(plane.distance(points) ** 2)) * 1000.0)


def measure_joint(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    camera_to_world: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: float = 0.002,
    iterations: int = 200,
    waypoints: int = 12,
    trim_percentile: float = 2.0,
    min_angle_deg: float = 25.0,
    expected_direction: Optional[np.ndarray] = None,
    max_tilt_deg: float = 35.0,
    rng: Optional[np.random.Generator] = None,
) -> JointMeasurement:
    """One depth image in, a weld path plus its quality signals out.

    The path is bounded by how much of the joint was actually seen, trimmed at both
    ends so a few stray points cannot stretch it past the real weld.
    """
    cloud = backproject(depth, intrinsics, mask)
    if len(cloud) < 6:
        raise RuntimeError("not enough valid depth samples to measure a joint")
    world = transform_points(cloud, camera_to_world)

    rng = rng or np.random.default_rng(0)
    face_a, inliers_a = fit_plane_ransac(world, threshold, iterations, rng)
    remaining = world[~inliers_a]
    if len(remaining) < 3:
        raise RuntimeError("only one surface in view; no joint to measure")
    face_b, inliers_b = fit_plane_ransac(remaining, threshold, iterations, rng)

    cosine = abs(float(face_a.normal @ face_b.normal))
    included = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    if included < min_angle_deg:
        # Two nearly parallel planes intersect in a line that shoots off to
        # infinity. Reporting a joint from that is worse than reporting nothing.
        raise RuntimeError(
            f"the two surfaces differ by only {included:.1f} degrees, which is "
            "not a joint; this view does not show both faces"
        )

    point, direction = intersect_planes(face_a, face_b)

    if expected_direction is not None:
        # The weld in this cell runs vertically. A fit that latches onto the table
        # and a wall also meets at ninety degrees, but its intersection is roughly
        # horizontal, and solving a horizontal line for a height sends the answer to
        # infinity. Knowing which way the weld runs is ordinary process knowledge.
        wanted = np.asarray(expected_direction, dtype=float)
        wanted = wanted / np.linalg.norm(wanted)
        tilt = np.degrees(np.arccos(np.clip(abs(float(direction @ wanted)), -1.0, 1.0)))
        if tilt > max_tilt_deg:
            raise RuntimeError(
                f"the fitted joint runs {tilt:.0f} degrees off the expected "
                "direction; these are not the two faces of the weld"
            )

    # How much of the joint this view actually covers.
    seen = np.vstack([world[inliers_a], remaining[inliers_b]])
    t = (seen - point) @ direction
    lo, hi = np.percentile(t, [trim_percentile, 100.0 - trim_percentile])
    path = point + np.outer(np.linspace(lo, hi, max(2, waypoints)), direction)

    return JointMeasurement(
        waypoints=path,
        point=point,
        direction=direction,
        face_a_inliers=int(inliers_a.sum()),
        face_b_inliers=int(inliers_b.sum()),
        face_a_rms_mm=_rms_mm(face_a, world[inliers_a]),
        face_b_rms_mm=_rms_mm(face_b, remaining[inliers_b]),
        included_angle_deg=included,
        path_length_mm=float(abs(hi - lo) * 1000.0),
    )


def project(
    points_world: np.ndarray, intrinsics: Intrinsics, camera_to_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """World points to pixel coordinates, with their depth along the optical axis."""
    points_world = np.atleast_2d(points_world)
    camera = transform_points(points_world, np.linalg.inv(camera_to_world))
    depth = -camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera[:, 0] * intrinsics.fx / depth + intrinsics.cx
        v = -camera[:, 1] * intrinsics.fy / depth + intrinsics.cy
    return np.stack([u, v], axis=1), depth


def region_of_interest(
    intrinsics: Intrinsics,
    centre_uv: np.ndarray,
    half_width_px: float,
    half_height_px: float,
) -> np.ndarray:
    """A rectangular pixel mask around an expected location.

    A depth image of a whole cell is mostly floor and fixture, and a plane fit over
    all of it locks onto the table. A real cell already knows roughly where the
    joint should be, because that is what the taught program says, so it looks
    there and refines. This is that crop, and it is a prior on position only: it
    says nothing about where the joint actually is inside the window.
    """
    cols, rows = np.meshgrid(np.arange(intrinsics.width), np.arange(intrinsics.height))
    return (np.abs(cols - centre_uv[0]) <= half_width_px) & (
        np.abs(rows - centre_uv[1]) <= half_height_px
    )


def working_volume(
    depth: np.ndarray,
    intrinsics: Intrinsics,
    centre_uv: np.ndarray,
    half_width_px: float,
    half_height_px: float,
    near_m: float,
    far_m: float,
) -> np.ndarray:
    """Pixels near the expected joint, both across the image and in range.

    A depth sensor has a working distance, and a cell knows roughly how far the
    joint should be. Without the range limit the fit happily takes the floor twenty
    metres away, which also meets a wall at ninety degrees and looks like a joint
    until you notice the weld it found is two metres long.
    """
    window = region_of_interest(intrinsics, centre_uv, half_width_px, half_height_px)
    in_range = np.isfinite(depth) & (depth >= near_m) & (depth <= far_m)
    return window & in_range
