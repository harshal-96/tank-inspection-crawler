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

# The part: a sheet-metal cabinet, width, height, depth, wall thickness.
PART_W = 0.90
PART_H = 1.20
PART_D = 0.45
PART_T = 0.02

# The weld joint runs down the corner where two panels meet, expressed in the
# part's own frame so it rotates with the part.
#
# This is where the OUTER faces meet, not where the centrelines do. A fillet weld
# on an outside corner is laid on the outer corner, and that is also what any
# sensor sees: fitting a plane to each visible face and intersecting them lands on
# the outer corner by construction. Defining it on the centreline instead puts a
# constant sqrt(2) * thickness / 2 offset, here 14.14 mm, into every measurement.
JOINT_LOCAL = (PART_W / 2, -(PART_H / 2))

# The bead sits proud of both panel faces so it is visible to the camera rather
# than buried in the corner. A bead centred on the crossing of two 20 mm panels
# shows only a 2 mm sliver on each face and photographs as a dark crevice; on the
# first pass 85 percent of it rendered black. A fillet weld on a sheet metal box
# lies ON the outside corner, so the bead is pushed outward along the diagonal.
BEAD_SIZE = (0.024, 0.024, PART_D + 0.006)
BEAD_Z = PART_D / 2 + 0.003
BEAD_PROUD = 0.008

# Ventilation slots. Real cut-outs, so they are gaps in the geometry and appear in
# depth as well as in the picture, exactly as they would on the part.
SLOT_WIDTH = 0.012
SLOT_HEIGHT = 0.185
SLOT_PITCH = 0.048
SLOT_CENTRE_Z = 0.245
SLOT_MARGIN = 0.075

# A weld bead is a row of overlapping solidified pools, which is what gives it the
# stack-of-dimes ripple. Pitch is the spacing between ripple crests; the fillet leg
# is how far the weld metal runs up each panel face.
RIPPLE_PITCH = 0.0035
FILLET_LEG = 0.008

# Torch stand-off from the joint along the outward diagonal.
STICKOUT_M = 0.012

# The inspection camera looks square on at the cabinet edge, from outside the
# corner along its diagonal, at the mid height of the edge. The whole weld joint
# is in shot, top to bottom, and the camera sees two large flat panel faces and
# nothing else: no rims, no horizontal surfaces for the plane fit to mistake for
# the fixture.
_EDGE_MID_Z = TABLE_TOP + PART_D / 2
_STANDOFF_M = 0.95
_DIAGONAL = 0.7071067811865476

CAMERA_AIM = (FIXTURE_X + PART_W / 2, -(PART_H / 2), _EDGE_MID_Z)
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


def slotted_face(
    length: float,
    height: float,
    thickness: float,
    along_axis: int,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """Break one face of the enclosure into the solid pieces around its slots.

    A row of vertical slots leaves a band below them, a band above them, a margin
    at each end and a pillar between each pair. Returned as box specs in the
    face's own frame, with the face lying along `along_axis` (0 for x, 1 for y)
    and its thickness on the other, height on z.

    Real cut-outs rather than dark paint: a slot is then a gap in the depth image
    too, which is what it would be on the part.
    """
    if along_axis not in (0, 1):
        raise ValueError("along_axis must be 0 (x) or 1 (y)")

    half = length / 2.0
    usable = length - 2.0 * SLOT_MARGIN
    count = max(0, int(usable // SLOT_PITCH))
    slot_bottom = SLOT_CENTRE_Z - SLOT_HEIGHT / 2.0
    slot_top = SLOT_CENTRE_Z + SLOT_HEIGHT / 2.0

    def spec(name, span, centre, z_span, z_centre):
        size = [0.0, 0.0, z_span]
        offset = [0.0, 0.0, z_centre]
        size[along_axis] = span
        size[1 - along_axis] = thickness
        offset[along_axis] = centre
        return (name, tuple(size), tuple(offset))

    pieces = [
        spec("sill", length, 0.0, slot_bottom, slot_bottom / 2.0),
        spec("header", length, 0.0, height - slot_top, (height + slot_top) / 2.0),
    ]
    if count == 0:
        pieces.append(spec("web", length, 0.0, SLOT_HEIGHT, SLOT_CENTRE_Z))
        return pieces

    # Slots centred on the face, with a pillar between each pair and a wider
    # margin at both ends.
    run = count * SLOT_PITCH
    first = -run / 2.0 + SLOT_PITCH / 2.0
    edges = [first + i * SLOT_PITCH for i in range(count)]

    left_margin = (edges[0] - SLOT_WIDTH / 2.0) - (-half)
    pieces.append(
        spec("end_a", left_margin, -half + left_margin / 2.0, SLOT_HEIGHT, SLOT_CENTRE_Z)
    )
    right_margin = half - (edges[-1] + SLOT_WIDTH / 2.0)
    pieces.append(
        spec("end_b", right_margin, half - right_margin / 2.0, SLOT_HEIGHT, SLOT_CENTRE_Z)
    )
    for index in range(count - 1):
        gap = edges[index + 1] - edges[index] - SLOT_WIDTH
        centre = (edges[index] + edges[index + 1]) / 2.0
        pieces.append(spec(f"pillar_{index}", gap, centre, SLOT_HEIGHT, SLOT_CENTRE_Z))
    return pieces
