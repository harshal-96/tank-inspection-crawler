"""Tests for geometric joint finding.

The scene is synthesised analytically, so the true joint position is known exactly
and the finder can be scored rather than eyeballed. No Isaac, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cell.depth_joint import (
    Intrinsics,
    Plane,
    backproject,
    find_joint,
    fit_plane,
    fit_plane_ransac,
    fit_two_planes,
    intersect_planes,
    joint_at_height,
    measure_joint,
    noisy_depth,
    project,
    transform_points,
    working_volume,
)

# A camera looking along world +y, with its own up axis along world +z, so a
# vertical joint in the world is vertical in the image. This mirrors the cell.
CAMERA_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
CAMERA_POSITION = np.array([0.0, -1.0, 0.5])

INTRINSICS = Intrinsics.from_camera(
    width=200, height=200, focal_length_mm=24.0, horizontal_aperture_mm=20.955
)

# Two faces meeting at a vertical line one metre in front of the camera.
ROOT_HALF = math.sqrt(2.0) / 2.0
FACE_A = Plane(normal=np.array([ROOT_HALF, 0.0, ROOT_HALF]), offset=-ROOT_HALF)
FACE_B = Plane(normal=np.array([-ROOT_HALF, 0.0, ROOT_HALF]), offset=-ROOT_HALF)

# In world coordinates that joint is the vertical line through (0, 0).
TRUE_JOINT_XY = (0.0, 0.0)
JOINT_HEIGHT = 0.5


def camera_to_world() -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = CAMERA_ROTATION
    matrix[:3, 3] = CAMERA_POSITION
    return matrix


def synthetic_depth(intrinsics: Intrinsics = INTRINSICS) -> np.ndarray:
    """Ray cast the two faces to make an exact depth image.

    A ray through pixel (u, v) has direction ((u-cx)/fx, -(v-cy)/fy, -1). Because
    the z component is -1, the ray parameter is itself the depth along the axis.
    """
    cols, rows = np.meshgrid(np.arange(intrinsics.width), np.arange(intrinsics.height))
    directions = np.stack(
        [
            (cols - intrinsics.cx) / intrinsics.fx,
            -(rows - intrinsics.cy) / intrinsics.fy,
            -np.ones_like(cols, dtype=float),
        ],
        axis=-1,
    )

    depth = np.full(directions.shape[:2], np.nan)
    for face in (FACE_A, FACE_B):
        denominator = directions @ face.normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = face.offset / denominator
        hit = np.isfinite(t) & (t > 0)
        # The visible surface is the nearer of the two.
        depth = np.where(hit & (np.isnan(depth) | (t < depth)), t, depth)
    return depth


# ----------------------------------------------------------------------- intrinsics


def test_intrinsics_from_camera():
    intrinsics = Intrinsics.from_camera(640, 640, 24.0, 20.955)
    assert intrinsics.fx == pytest.approx(24.0 / 20.955 * 640)
    assert intrinsics.fx == pytest.approx(intrinsics.fy)  # square aperture
    assert (intrinsics.cx, intrinsics.cy) == (320.0, 320.0)


def test_non_square_aperture_gives_different_focal_scales():
    intrinsics = Intrinsics.from_camera(640, 480, 24.0, 20.955, 15.2908)
    assert intrinsics.fx != pytest.approx(intrinsics.fy)


# ---------------------------------------------------------------------- projection


def test_backproject_centre_pixel_lies_on_the_axis():
    depth = np.full((INTRINSICS.height, INTRINSICS.width), np.nan)
    depth[INTRINSICS.height // 2, INTRINSICS.width // 2] = 2.0
    cloud = backproject(depth, INTRINSICS)
    assert len(cloud) == 1
    assert cloud[0] == pytest.approx([0.0, 0.0, -2.0])


def test_backproject_skips_invalid_samples():
    depth = np.full((INTRINSICS.height, INTRINSICS.width), np.nan)
    depth[10, 10] = 1.0
    depth[20, 20] = 0.0  # a zero reading is no reading
    depth[30, 30] = -1.0
    assert len(backproject(depth, INTRINSICS)) == 1


def test_backproject_honours_a_mask():
    depth = np.ones((INTRINSICS.height, INTRINSICS.width))
    mask = np.zeros_like(depth, dtype=bool)
    mask[:5, :5] = True
    assert len(backproject(depth, INTRINSICS, mask)) == 25


def test_backproject_rejects_mismatched_shape():
    with pytest.raises(ValueError):
        backproject(np.ones((10, 10)), INTRINSICS)


def test_transform_points_applies_rotation_and_translation():
    points = np.array([[0.0, 0.0, -1.0]])
    world = transform_points(points, camera_to_world())
    assert world[0] == pytest.approx([0.0, 0.0, 0.5])


# --------------------------------------------------------------------------- fits


def test_fit_plane_recovers_a_known_plane():
    rng = np.random.default_rng(0)
    points = np.stack(
        [rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400), np.full(400, 0.25)], axis=1
    )
    plane = fit_plane(points)
    assert abs(plane.normal[2]) == pytest.approx(1.0, abs=1e-6)
    assert abs(plane.offset) == pytest.approx(0.25, abs=1e-6)


def test_fit_plane_needs_three_points():
    with pytest.raises(ValueError):
        fit_plane(np.zeros((2, 3)))


def test_ransac_ignores_outliers():
    rng = np.random.default_rng(1)
    inliers = np.stack(
        [rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500), np.zeros(500)], axis=1
    )
    outliers = rng.uniform(-1, 1, size=(120, 3)) + np.array([0, 0, 3.0])
    plane, mask = fit_plane_ransac(np.vstack([inliers, outliers]), threshold=0.01, rng=rng)

    assert abs(plane.normal[2]) == pytest.approx(1.0, abs=1e-3)
    assert mask[:500].sum() > 480  # kept the real surface
    assert mask[500:].sum() == 0  # rejected every outlier


def test_two_planes_rejects_a_single_flat_surface():
    rng = np.random.default_rng(2)
    flat = np.stack(
        [rng.uniform(-1, 1, 600), rng.uniform(-1, 1, 600), np.zeros(600)], axis=1
    )
    # Either rejection is correct: RANSAC may claim the whole surface, leaving
    # nothing for a second plane, or fit both and find them parallel.
    with pytest.raises(RuntimeError, match="no joint to find|not a joint"):
        fit_two_planes(flat, threshold=0.005, rng=rng)


# ------------------------------------------------------------------- intersection


def test_intersect_planes_recovers_a_known_line():
    a = Plane(np.array([1.0, 0.0, 0.0]), 2.0)  # x = 2
    b = Plane(np.array([0.0, 1.0, 0.0]), 3.0)  # y = 3
    point, direction = intersect_planes(a, b)
    assert point[0] == pytest.approx(2.0)
    assert point[1] == pytest.approx(3.0)
    assert abs(direction[2]) == pytest.approx(1.0)


def test_parallel_planes_have_no_line():
    a = Plane(np.array([0.0, 0.0, 1.0]), 1.0)
    b = Plane(np.array([0.0, 0.0, 1.0]), 2.0)
    with pytest.raises(ValueError):
        intersect_planes(a, b)


def test_joint_at_height_walks_the_line():
    point = np.array([1.0, 2.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    assert joint_at_height(point, direction, 5.0) == pytest.approx((1.0, 2.0))


def test_joint_at_height_rejects_a_horizontal_line():
    with pytest.raises(ValueError):
        joint_at_height(np.zeros(3), np.array([1.0, 0.0, 0.0]), 1.0)


# -------------------------------------------------------------------- end to end


def test_finds_the_joint_on_clean_depth():
    joint = find_joint(
        synthetic_depth(), INTRINSICS, camera_to_world(), at_height=JOINT_HEIGHT
    )
    error_mm = 1000.0 * math.dist(joint, TRUE_JOINT_XY)
    assert error_mm < 0.5, f"clean depth should be near exact, got {error_mm:.3f} mm"


def test_accuracy_degrades_gracefully_with_sensor_noise():
    """The number that matters: how far off the joint is at realistic noise levels."""
    truth = np.array(TRUE_JOINT_XY)
    errors = {}
    for sigma_mm in (0.0, 0.5, 1.0, 2.0):
        rng = np.random.default_rng(7)
        depth = noisy_depth(synthetic_depth(), sigma_mm / 1000.0, rng)
        joint = find_joint(
            depth,
            INTRINSICS,
            camera_to_world(),
            at_height=JOINT_HEIGHT,
            threshold=max(0.002, 4 * sigma_mm / 1000.0),
            rng=rng,
        )
        errors[sigma_mm] = 1000.0 * float(np.linalg.norm(np.array(joint) - truth))

    assert errors[0.0] < 0.5
    # Plane fitting averages over thousands of samples, so range noise should not
    # translate into joint error one for one.
    assert errors[2.0] < 5.0, f"2 mm of range noise gave {errors[2.0]:.2f} mm of error"
    assert errors[0.0] <= errors[2.0]


def test_survives_dropout():
    rng = np.random.default_rng(11)
    depth = noisy_depth(synthetic_depth(), 0.0005, rng, dropout=0.3)
    joint = find_joint(
        depth, INTRINSICS, camera_to_world(), at_height=JOINT_HEIGHT, rng=rng
    )
    assert 1000.0 * math.dist(joint, TRUE_JOINT_XY) < 5.0


def test_joint_height_does_not_change_a_vertical_joint():
    """A vertical joint reads the same at any height, which is the invariant the
    correction relies on."""
    depth = synthetic_depth()
    low = find_joint(depth, INTRINSICS, camera_to_world(), at_height=0.2)
    high = find_joint(depth, INTRINSICS, camera_to_world(), at_height=0.8)
    assert low == pytest.approx(high, abs=1e-6)


def test_refuses_an_empty_depth_image():
    with pytest.raises(RuntimeError):
        find_joint(
            np.full((INTRINSICS.height, INTRINSICS.width), np.nan),
            INTRINSICS,
            camera_to_world(),
            at_height=JOINT_HEIGHT,
        )


# ------------------------------------------------------------------------ guards


def test_measure_rejects_a_near_parallel_pair():
    """Two nearly parallel planes meet at infinity; reporting that is worse than
    reporting nothing."""
    depth = synthetic_depth()
    with pytest.raises(RuntimeError, match="not a joint"):
        measure_joint(depth, INTRINSICS, camera_to_world(), min_angle_deg=120.0)


def test_measure_rejects_a_joint_running_the_wrong_way():
    """A table meeting a wall is also square, but its intersection is horizontal."""
    depth = synthetic_depth()
    with pytest.raises(RuntimeError, match="off the expected direction"):
        measure_joint(
            depth,
            INTRINSICS,
            camera_to_world(),
            expected_direction=np.array([1.0, 0.0, 0.0]),
        )


def test_measure_accepts_the_direction_it_expects():
    measurement = measure_joint(
        synthetic_depth(),
        INTRINSICS,
        camera_to_world(),
        expected_direction=np.array([0.0, 0.0, 1.0]),
    )
    assert measurement.at_height(JOINT_HEIGHT) == pytest.approx(TRUE_JOINT_XY, abs=5e-4)


def test_measurement_reports_the_path_and_its_signals():
    measurement = measure_joint(synthetic_depth(), INTRINSICS, camera_to_world())
    assert measurement.waypoints.shape == (12, 3)
    assert measurement.included_angle_deg == pytest.approx(90.0, abs=0.5)
    assert measurement.inlier_balance > 0.9
    assert set(measurement.summary()) >= {"included_angle_deg", "inlier_balance",
                                          "path_length_mm", "waypoints"}


def test_inlier_balance_is_zero_when_a_face_is_unseen():
    from cell.depth_joint import JointMeasurement

    lopsided = JointMeasurement(
        waypoints=np.zeros((2, 3)), point=np.zeros(3), direction=np.array([0, 0, 1.0]),
        face_a_inliers=9000, face_b_inliers=0, face_a_rms_mm=0.1, face_b_rms_mm=0.1,
        included_angle_deg=90.0, path_length_mm=400.0,
    )
    assert lopsided.inlier_balance == 0.0


def test_working_volume_excludes_what_is_out_of_range():
    depth = synthetic_depth()
    centre = np.array([INTRINSICS.cx, INTRINSICS.cy])
    wide = working_volume(depth, INTRINSICS, centre, 500, 500, 0.0, 100.0)
    banded = working_volume(depth, INTRINSICS, centre, 500, 500, 0.9, 1.1)
    assert banded.sum() < wide.sum()
    assert banded.sum() > 0


def test_working_volume_excludes_what_is_off_to_the_side():
    depth = synthetic_depth()
    centre = np.array([INTRINSICS.cx, INTRINSICS.cy])
    assert working_volume(depth, INTRINSICS, centre, 10, 10, 0.0, 100.0).sum() < 500


def test_project_round_trips_through_backprojection():
    """A point projected to a pixel, then back out at its own depth, returns."""
    world = np.array([[0.05, -0.30, 0.62]])
    uv, distance = project(world, INTRINSICS, camera_to_world())
    depth = np.full((INTRINSICS.height, INTRINSICS.width), np.nan)
    depth[int(round(uv[0][1])), int(round(uv[0][0]))] = distance[0]
    back = transform_points(backproject(depth, INTRINSICS), camera_to_world())
    assert back[0] == pytest.approx(world[0], abs=3e-3)
