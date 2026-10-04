"""The wall thickness model, with no Blender in it.

This lived inside the capture script, which imports bpy and therefore only
runs inside Blender. That made the one piece of physics in the tank half
unreachable from a test, from a notebook, or from anything that wanted to ask
what the wall is without opening a 3D application first.

It is pure arithmetic over the planted patches. Thinning is stored as
parameters rather than as a grid, so the thickness can be asked at any point
on the shell and not only at the points somebody happened to sample, and the
answer is the same number the capture wrote into the manifest.
"""

from __future__ import annotations

import math

# Read from the builder rather than copied, for the same reason the adhesion
# budget does it: two copies of a course thickness means one going stale.
from tank.adhesion import scene_constants

_SCENE = scene_constants("COURSE_MM", "COURSE_H", "COURSES", "RADIUS")
COURSE_MM = _SCENE["COURSE_MM"]
COURSE_H = _SCENE["COURSE_H"]
COURSES = _SCENE["COURSES"]
RADIUS = _SCENE["RADIUS"]


def nominal_mm(z: float) -> float:
    """Plate thickness of the course this height sits in."""
    course = min(COURSES - 1, max(0, int(z // COURSE_H)))
    return COURSE_MM[course]


def thickness_mm(defects, theta: float, z: float) -> tuple[float, list[str]]:
    """True wall thickness at a point, and which patches took metal off it.

    Distance is measured along the wall rather than through the tank, because
    a patch is a patch of surface and the chord would quietly make every
    distance shorter than it is.
    """
    left = nominal_mm(z)
    took: list[str] = []
    for d in defects:
        if d.get("kind") != "thinning":
            continue
        d_theta = ((theta - d["theta_rad"] + math.pi) % math.tau) - math.pi
        along = d_theta * RADIUS
        dist = math.hypot(along, z - d["z_m"])
        if dist < d["patch_radius_m"]:
            # Cosine falloff, so a patch has a soft edge rather than a step.
            # A step would make the probe planner's job artificially easy: it
            # could find an edge by bisection instead of having to look.
            left -= d["loss_mm"] * math.cos(math.pi / 2 * dist / d["patch_radius_m"])
            took.append(d["id"])
    return round(left, 3), took
