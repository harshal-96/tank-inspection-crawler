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

# The weld joint runs down the corner where two panels meet. Expressed in the
# part's own frame, so it rotates with the part.
JOINT_LOCAL = (PART_W / 2 - PART_T / 2, -(PART_H / 2 - PART_T / 2))

# The bead sits proud of both panel faces so it is visible to the camera rather
# than buried in the corner. A bead centred on the crossing of two 20 mm panels
# shows only a 2 mm sliver on each face and photographs as a dark crevice; on the
# first pass 85 percent of it rendered black. A fillet weld on a sheet metal box
# lies ON the outside corner, so the bead is pushed outward along the diagonal.
BEAD_SIZE = (0.024, 0.024, PART_D + 0.006)
BEAD_Z = PART_D / 2 + 0.003
BEAD_PROUD = 0.008

# A weld bead is a row of overlapping solidified pools, which is what gives it the
# stack-of-dimes ripple. Pitch is the spacing between ripple crests; the fillet leg
# is how far the weld metal runs up each panel face.
RIPPLE_PITCH = 0.0035
FILLET_LEG = 0.008

# Torch stand-off from the joint along the outward diagonal.
STICKOUT_M = 0.012

# The inspection camera. Framing was chosen empirically, not from a projection
# calculation, because the two disagree at this angle.
CAMERA_EYE = (4.75, -1.45, 1.25)
CAMERA_AIM = (FIXTURE_X + JOINT_LOCAL[0], JOINT_LOCAL[1], TABLE_TOP + 0.2)


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
