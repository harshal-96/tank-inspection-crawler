"""Cell geometry and the forward model from part pose to joint position.

Everything here is plain arithmetic with no simulation or vision dependency, so it
can be imported by the Isaac process, the service and the tests alike.

Coordinates are the robot frame, in metres. x runs down the line, y is lateral,
z is up.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- cell

FIXTURE_X = 3.95   # where the part comes to rest against the stop
TABLE_TOP = 0.75   # fixture table height

# The part: an open-topped tray folded from one flat blank. The blank is cut with
# a square notch at each corner so the four sides fold up without fouling each
# other. PART_W and PART_H are the base the sides were folded from, PART_D is how
# far each side stands above it, PART_T is the sheet.
#
# Two millimetres, because this is sheet metal folded on a press brake. Plate that
# thick enough to matter would not be folded at all, it would be welded up from
# cut pieces, and none of what follows would apply to it.
PART_W = 0.90
PART_H = 1.20
PART_D = 0.45
PART_T = 0.002

# Each side leans outward by the sheet thickness, so the outside of the tray is a
# thickness larger than the base on every face and a thickness taller than the
# side height.
OUTER_W = PART_W + 2 * PART_T
OUTER_H = PART_H + 2 * PART_T
OUTER_D = PART_D + PART_T

# Where the torch goes, expressed in the part's own frame so it rotates with it.
#
# This corner is an open joint. Each side is folded up off its own fold line and
# stops where the corner notch was cut, so the two sides touch along the line
# where the fold lines cross, and outboard of that line there is a small open
# square the thickness of the sheet on each side. The weld is laid on the outside
# and fills it, so the torch aims at the outer corner: where the two outer faces
# would meet if they ran on.
#
# That is also the only point a sensor can recover. Fitting a line to each visible
# outer face and intersecting them lands there by construction. There is no metal
# at the point itself, which is the nature of an open corner, and does not stop it
# being found to a few tenths of a millimetre.
JOINT_LOCAL = (OUTER_W / 2, -(OUTER_H / 2))

# The bead sits proud of both panel faces so it is visible to the camera rather
# than buried in the corner. A bead centred on the crossing of two 20 mm panels
# shows only a 2 mm sliver on each face and photographs as a dark crevice; on the
# first pass 85 percent of it rendered black. A fillet weld on a sheet metal box
# lies ON the outside corner, so the bead is pushed outward along the diagonal.
BEAD_SIZE = (0.006, 0.006, OUTER_D + 0.002)
BEAD_Z = OUTER_D / 2 + 0.001
BEAD_PROUD = 0.002

# A weld bead is a row of overlapping solidified pools, which is what gives it the
# stack-of-dimes ripple. Pitch is the spacing between ripple crests; the fillet leg
# is how far the weld metal runs up each panel face.
RIPPLE_PITCH = 0.002
FILLET_LEG = 0.003

# Torch stand-off from the joint along the outward diagonal.
STICKOUT_M = 0.008

# The inspection camera looks square on at the cabinet edge, from outside the
# corner along its diagonal, at the mid height of the edge. The whole weld joint
# is in shot, top to bottom, and the camera sees two large flat panel faces and
# nothing else: no rims, no horizontal surfaces for the plane fit to mistake for
# the fixture.
_EDGE_MID_Z = TABLE_TOP + OUTER_D / 2
_STANDOFF_M = 0.95
_DIAGONAL = 0.7071067811865476

CAMERA_AIM = (FIXTURE_X + JOINT_LOCAL[0], JOINT_LOCAL[1], _EDGE_MID_Z)
CAMERA_EYE = (
    CAMERA_AIM[0] + _STANDOFF_M * _DIAGONAL,
    CAMERA_AIM[1] - _STANDOFF_M * _DIAGONAL,
    _EDGE_MID_Z + 0.06,
)


def nominal_joint() -> tuple[float, float]:
    """Where a taught program believes the joint is: the part arriving perfectly."""
    return (FIXTURE_X + JOINT_LOCAL[0], JOINT_LOCAL[1])


def true_joint(part_x: float, part_y: float, part_yaw: float) -> tuple[float, float]:
    """Where the joint actually is, given the pose the part settled into.

    This is the ground truth the detector is scored against. It is only knowable
    in simulation, which is the whole reason the validation happens there.

    part_yaw is in radians.
    """
    cos_a, sin_a = math.cos(part_yaw), math.sin(part_yaw)
    local_x, local_y = JOINT_LOCAL
    return (
        part_x + local_x * cos_a - local_y * sin_a,
        part_y + local_x * sin_a + local_y * cos_a,
    )


def seam_ends(part_x: float, part_y: float, part_yaw: float) -> tuple:
    """The two ends of the seam on a perfectly formed part, in world coordinates.

    This is what a taught program drives between: a straight line from the bottom
    of the corner to the top. On a real part the seam is neither straight nor in
    this place, which is the whole problem.
    """
    joint_x, joint_y = true_joint(part_x, part_y, part_yaw)
    bottom = (joint_x, joint_y, TABLE_TOP + PART_T)
    top = (joint_x, joint_y, TABLE_TOP + PART_T + PART_D)
    return bottom, top


def true_seam(
    part_x: float,
    part_y: float,
    part_yaw: float,
    corner,
    samples: int = 25,
) -> list[tuple[float, float, float]]:
    """Where the seam actually runs, bottom to top, given the forming error.

    The two sides meeting at this corner each lean and bow by their own amount, so
    the seam they make is a curve in space rather than the vertical line the part
    drawing shows. This is the ground truth a tracked weld is scored against, and
    it is only knowable in simulation, which is why the scoring happens there.

    corner is a cell.forming.CornerForming. Passed in rather than imported so this
    module stays free of every other part of the cell.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2")

    cos_a, sin_a = math.cos(part_yaw), math.sin(part_yaw)
    local_x, local_y = JOINT_LOCAL
    points = []
    for i in range(samples):
        up = PART_D * i / (samples - 1)          # height above the fold
        shift_x, shift_y = corner.seam_offset_m(up, PART_D)
        lx, ly = local_x + shift_x, local_y + shift_y
        points.append(
            (
                part_x + lx * cos_a - ly * sin_a,
                part_y + lx * sin_a + ly * cos_a,
                TABLE_TOP + PART_T + up,
            )
        )
    return points


def taught_miss_mm(
    part_x: float,
    part_y: float,
    part_yaw: float,
    corner,
    samples: int = 25,
) -> float:
    """Worst distance from the taught straight path to the seam it should follow.

    The taught path is the line between the seam ends of a nominal part sitting
    where the fixture says it will. Everything that moves the real seam away from
    that line shows up here: how the part arrived, and how the press formed it.
    """
    nominal_x, nominal_y = nominal_joint()
    bottom = (nominal_x, nominal_y, TABLE_TOP + PART_T)
    top = (nominal_x, nominal_y, TABLE_TOP + PART_T + PART_D)

    axis = tuple(top[i] - bottom[i] for i in range(3))
    length = math.sqrt(sum(c * c for c in axis))
    unit = tuple(c / length for c in axis)

    worst = 0.0
    for point in true_seam(part_x, part_y, part_yaw, corner, samples):
        rel = tuple(point[i] - bottom[i] for i in range(3))
        along = sum(rel[i] * unit[i] for i in range(3))
        off = tuple(rel[i] - along * unit[i] for i in range(3))
        worst = max(worst, math.sqrt(sum(c * c for c in off)))
    return 1000.0 * worst


def torch_target(joint_x: float, joint_y: float, part_yaw: float = 0.0) -> tuple[float, float]:
    """Torch tip position: outboard of the joint along the part's outward diagonal."""
    local_x, local_y = JOINT_LOCAL
    norm = math.hypot(local_x, local_y)
    dir_x, dir_y = local_x / norm, local_y / norm

    cos_a, sin_a = math.cos(part_yaw), math.sin(part_yaw)
    out_x = dir_x * cos_a - dir_y * sin_a
    out_y = dir_x * sin_a + dir_y * cos_a
    return (joint_x + out_x * STICKOUT_M, joint_y + out_y * STICKOUT_M)


def distance_mm(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Planar distance in millimetres."""
    return 1000.0 * math.hypot(a[0] - b[0], a[1] - b[1])


def blind_error_mm(part_x: float, part_y: float, part_yaw: float) -> float:
    """How far a taught program would miss by, with no correction at all.

    This is the baseline the corrected number is compared against.
    """
    return distance_mm(nominal_joint(), true_joint(part_x, part_y, part_yaw))


def orbit_about_joint(
    part_x: float,
    part_y: float,
    part_yaw: float,
    joint_x: float,
    joint_y: float,
    theta: float,
) -> tuple[float, float, float]:
    """Swing the part around its own joint, to present a different face to a camera.

    The cell's camera cannot be moved once the scene is composed, so a different
    viewing angle is produced by turning the part instead. Turning it about the
    joint axis specifically means the joint itself does not move: the ground truth
    is identical from every view, so any difference in the measurement is the
    perception changing, not the target.

    All angles in radians. Returns the new part pose.
    """
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    offset_x, offset_y = part_x - joint_x, part_y - joint_y
    return (
        joint_x + offset_x * cos_t - offset_y * sin_t,
        joint_y + offset_x * sin_t + offset_y * cos_t,
        part_yaw + theta,
    )
