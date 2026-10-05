"""What was in front of the lens at every station, worked out without Blender.

The answer key used to be computed inside Blender at capture time and frozen
into the manifest, which left two problems. Nobody could re-derive it without
the renderer, and it projected every fault as if it sat on the crown of the
bead. Most of them do not. A crack is drawn a third of the way up the bead and
shows on the plate above and below it, and near the edge of a wide lens that
difference in depth moves a point by twenty pixels. At station 12 it moved the
whole crack out of the key while the crack itself was plainly in the picture,
and the inspector, which passed that frame, was marked right for missing it.

So the key is now computed here, from two things a reader can check:

- the camera, rebuilt from the same constants the scene builder uses, and
  checked against Blender's own projection of 630 wall points in
  projection_check.json;
- where each fault was drawn. For the seam faults on the run that is the
  vertices of the rendered meshes themselves, read out of the evaluation scene
  into fault_geometry.json. Patches of wall loss keep the description from the
  scene file, projected at the depth their rust is drawn.

Nothing here imports bpy, so the scorer, the dashboard and the tests all use
the same rule, and none of them needs a GPU.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from tank.adhesion import scene_constants

ROOT = Path(__file__).resolve().parent.parent
SURVEY = ROOT / "tank" / "data" / "survey"
GEOMETRY = SURVEY / "fault_geometry.json"
GEOMETRY_NOTE = (
    "The vertices of every fault mesh on the survey run, read out of the "
    "evaluation scene the frames were rendered from. Wall coordinates: theta in "
    "radians, z in metres, and inset, how far toward the middle of the tank the "
    "vertex is drawn. tank/truth.py projects these, so the answer key is the "
    "geometry that was rendered and not a description of it.")

_C = scene_constants("RADIUS", "BEAD_R", "RUST_INSET", "CAMERA_EYE", "CAMERA_LENS_MM")
RADIUS = _C["RADIUS"]
BEAD_R = _C["BEAD_R"]
RUST_INSET = _C["RUST_INSET"]
CAMERA_EYE = _C["CAMERA_EYE"]
LENS_MM = _C["CAMERA_LENS_MM"]
# Blender's default, which the builder never changes. projection_check.json is
# the evidence: a different sensor would not reproduce Blender's numbers.
SENSOR_MM = 36.0
RESOLUTION = (1280, 720)

# Worst first: the order a fault would stop the job. The inspector's prompt is
# written from this same tuple (tank/inspect_seam.py), so the scorer cannot
# rank by an order the inspector was never told.
SEVERITY = ("crack", "burn_through", "lack_of_fusion", "undercut",
            "porosity", "corrosion", "none")

# What a camera calls a fault the scene names by its cause.
CAMERA_WORD = {"thinning": "corrosion"}

# A fault has to fill this share of itself or of the frame to count as being
# in the picture. Below it the frame holds a sliver at the very edge, which an
# inspector may name or not and is not marked down either way.
TRACE = 0.08

# How wide each kind of fault is, half width in metres, across the line it
# runs along. One shared figure made a crack as wide as the bead.
HALF_WIDTH_M = {"crack": 0.002, "lack_of_fusion": 0.006, "undercut": 0.006,
                "burn_through": 0.009, "porosity": 0.012}


# ------------------------------------------------------------------ the camera


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit(a):
    n = math.sqrt(_dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


def wall_point(theta, z, inset=0.0):
    """A point on the shell, inset metres toward the middle of the tank."""
    r = RADIUS - inset
    return (r * math.cos(theta), r * math.sin(theta), z)


@lru_cache(maxsize=256)
def camera_pose(theta: float, z: float):
    """Where the lens is and which way it faces, with the crawler at a station.

    The crawler's frame is +X along the seam, +Y down the wall, +Z off the
    plate (blender_tank.crawler_matrix), and the camera is aimed straight at
    the plate from CAMERA_EYE with +Y as up (build_crawler). Returned as an
    origin and the camera's right, up and back axes in world space.
    """
    tangent = (-math.sin(theta), math.cos(theta), 0.0)
    normal = (-math.cos(theta), -math.sin(theta), 0.0)
    across = _cross(normal, tangent)
    origin_on_wall = wall_point(theta, z)

    def to_world(v, point=False):
        w = tuple(tangent[i] * v[0] + across[i] * v[1] + normal[i] * v[2]
                  for i in range(3))
        return tuple(w[i] + origin_on_wall[i] for i in range(3)) if point else w

    eye = CAMERA_EYE
    target = (eye[0], eye[1], 0.0)
    forward = _unit(tuple(target[i] - eye[i] for i in range(3)))
    right = _unit(_cross(forward, (0.0, 1.0, 0.0)))
    up = _cross(right, forward)
    back = tuple(-f for f in forward)
    return (to_world(eye, point=True), to_world(right), to_world(up), to_world(back))


def project(world, theta: float, z: float):
    """The same numbers as Blender's world_to_camera_view: x and y across the
    frame from its bottom left corner, and depth along the lens axis."""
    origin, right, up, back = camera_pose(theta, z)
    rel = tuple(world[i] - origin[i] for i in range(3))
    depth = -_dot(rel, back)
    if depth <= 0.0:
        return (0.5, 0.5, depth)
    half_x = SENSOR_MM / 2 / LENS_MM
    # Auto sensor fit on a landscape frame fits the width.
    half_y = half_x * RESOLUTION[1] / RESOLUTION[0]
    x = 0.5 + _dot(rel, right) / (2 * depth * half_x)
    y = 0.5 + _dot(rel, up) / (2 * depth * half_y)
    return (x, y, depth)


def in_frame(xyd) -> bool:
    x, y, d = xyd
    return d > 0 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


# ------------------------------------------------------------------ the faults


@lru_cache(maxsize=1)
def drawn_geometry() -> dict:
    """The rendered vertices of the seam faults on the run, by fault id."""
    if not GEOMETRY.exists():
        return {}
    data = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    return {fid: [tuple(p) for p in pts] for fid, pts in data["faults"].items()}


def footprint(d) -> list[tuple[float, float, float]]:
    """Points spread over the whole of a fault: theta, z and the depth it is
    drawn at. The truth has to cover the thing, not point at the middle of it.
    """
    drawn = drawn_geometry().get(d["id"])
    if drawn:
        return drawn
    theta, z = d["theta_rad"], d["z_m"]
    if d["kind"] == "thinning":               # a patch, and islands beyond it
        r = d["patch_radius_m"]
        points = [(theta, z, RUST_INSET)]
        for k in range(12):
            a = math.tau * k / 12
            for reach in (0.6, 1.0, 1.7):
                points.append((theta + math.cos(a) * r * reach / RADIUS,
                               z + math.sin(a) * r * reach, RUST_INSET))
        points.extend((it, iz, RUST_INSET)
                      for it, iz, _ in (d.get("rust_islands") or []))
        return points
    # Seam faults off the run were never rendered in detail and never come
    # near a frame; a line along or across the seam is enough to say so.
    if "arc_from" in d:
        lo, hi = d["arc_from"], d["arc_to"]
        return [(lo + (hi - lo) * i / 8, z, BEAD_R) for i in range(9)]
    if "length_mm" in d:
        half = d["length_mm"] / 2000
        return [(theta, z - half + 2 * half * i / 6, BEAD_R) for i in range(7)]
    if d["kind"] == "porosity":
        return [(theta + (i - 2) * 0.014 / RADIUS, z, BEAD_R) for i in range(5)]
    return [(theta, z, BEAD_R)]


def covers(d):
    """A test for whether a point on the wall is inside this fault."""
    theta, z, R = d["theta_rad"], d["z_m"], RADIUS
    hw = HALF_WIDTH_M.get(d["kind"], 0.010)
    if "arc_from" in d:                       # runs along a circumferential seam
        lo, hi = d["arc_from"], d["arc_to"]
        return lambda t, y: lo <= t <= hi and abs(y - z) <= hw
    if d["kind"] == "thinning":               # a patch, and its islands
        r = d["patch_radius_m"] * 0.70        # the mean reach of the ragged edge
        islands = d.get("rust_islands") or []

        def rusted(t, y):
            if math.hypot((t - theta) * R, y - z) <= r:
                return True
            return any(math.hypot((t - it) * R, y - iz) <= ir
                       for it, iz, ir in islands)

        return rusted
    if "length_mm" in d:                      # runs up a vertical seam
        half = d["length_mm"] / 2000
        return lambda t, y: abs(y - z) <= half and abs((t - theta) * R) <= hw
    if d["kind"] == "porosity":               # a short cluster of pores
        return lambda t, y: abs((t - theta) * R) <= 0.035 and abs(y - z) <= hw
    return lambda t, y: math.hypot((t - theta) * R, y - z) <= 0.02


@lru_cache(maxsize=512)
def wall_in_frame(centre_theta: float, centre_z: float, inset: float):
    """A grid of wall points the lens can see, at one depth.

    Twenty millimetre steps. At thirty the grid walked over rust islands small
    enough to fit between two samples and reported clean plate.
    """
    shown = []
    for i in range(37):
        t = centre_theta + (i - 18) * 0.020 / RADIUS
        for j in range(25):
            y = centre_z + (j - 12) * 0.016
            if in_frame(project(wall_point(t, y, inset), centre_theta, centre_z)):
                shown.append((t, y))
    return tuple(shown)


def faults_in_view(defects, centre_theta: float, centre_z: float) -> list[dict]:
    """Every fault the lens is pointing at, and how much of the picture it is."""
    seen = []
    for d in defects:
        points = footprint(d)
        inside = []
        for theta, z, inset in points:
            xyd = project(wall_point(theta, z, inset), centre_theta, centre_z)
            if in_frame(xyd):
                inside.append(xyd)
        depth = sorted(p[2] for p in points)[len(points) // 2]
        shown = wall_in_frame(centre_theta, centre_z, round(depth, 6))
        here = covers(d)
        share = (sum(1 for t, y in shown if here(t, y)) / len(shown)) if shown else 0.0
        if not inside and share <= 0.0:
            continue
        if inside:
            x, y, rng_m = inside[len(inside) // 2]
            u, v = float(x), float(1.0 - y)
        else:
            u, v, rng_m = 0.5, 0.5, 0.0
        seen.append({"id": d["id"], "kind": d["kind"],
                     "u": round(u, 4), "v": round(v, 4),
                     "range_m": round(rng_m, 4),
                     "of_the_fault": round(len(inside) / len(points), 3),
                     "of_the_frame": round(share, 3),
                     "coverage": round(max(len(inside) / len(points), share), 3),
                     "footprint": "drawn" if d["id"] in drawn_geometry() else "described",
                     "visible_to_a_camera": d["visible"],
                     "rejectable": bool(d.get("rejectable"))})
    return seen


# ------------------------------------------------------------------- the key


def worst(kinds) -> str:
    """The fault that would stop the job first, out of everything in one frame."""
    ranked = sorted((k for k in kinds if k in SEVERITY), key=SEVERITY.index)
    return ranked[0] if ranked else "none"


def truth_for(frame: dict) -> tuple[str, list[str], list[str]]:
    """What was in front of the lens, the worst of it, and what merely grazed it.

    Only faults a camera could see count against the picture. A patch of wall
    loss with nothing on the surface is in the frame in the sense that the
    crawler was pointing at it, and no camera on earth would find it, so
    marking the model down for missing it would be marking it down for physics.
    """
    visible = [f for f in frame["faults_in_view"] if f["visible_to_a_camera"]]
    seen = [CAMERA_WORD.get(f["kind"], f["kind"]) for f in visible
            if f.get("coverage", 1.0) >= TRACE]
    grazed = [CAMERA_WORD.get(f["kind"], f["kind"]) for f in visible
              if f.get("coverage", 1.0) < TRACE]
    return worst(seen), seen, grazed


def load_answers(path: Path, model: str | None = None) -> dict[int, dict]:
    """Recorded verdicts by station, from a scorer's jsonl.

    One verdict a station. A file holding two for the same station and model
    would let a reader keep whichever one scored better, so it is refused
    rather than quietly resolved by taking the first.
    """
    out: dict[int, dict] = {}
    seen: set[tuple] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if model is not None and row.get("model") not in (None, model):
                continue
            key = (row.get("model"), row["station"])
            if key in seen:
                raise ValueError(f"{path}: line {number} is a second verdict for "
                                 f"station {row['station']} from {row.get('model')}")
            seen.add(key)
            out.setdefault(row["station"], row)
    return out
