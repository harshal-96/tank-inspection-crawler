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
  vertices of the rendered meshes themselves, and for every patch of rust a
  camera can see it is the rendered edge of the patch and of each island,
  all read out of the evaluation scene into fault_geometry.json. Wall loss
  with nothing on the surface keeps the description from the scene file.

How much of the picture a fault fills is measured on the picture: a grid of
points across the frame, each cast onto the wall and tested against the
fault. It used to be a grid of wall points round the crawler's own origin,
and the lens sits 0.46 m ahead of that, so the grid covered the left sixth of
every frame and a rust patch filling the right edge of frame 2 came out as a
sliver.

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
    "vertex is drawn. outlines holds the rendered edge of every visible rust "
    "patch and of each of its islands, as closed rings. tank/truth.py projects "
    "these, so the answer key is the geometry that was rendered and not a "
    "description of it.")

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

# The picture is sampled on this grid, columns by rows, to measure how much of
# it a fault fills. 20 pixels a sample on a 1280 by 720 frame.
FRAME_SAMPLES = (64, 36)

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
def _geometry_file() -> dict:
    if not GEOMETRY.exists():
        return {}
    return json.loads(GEOMETRY.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def drawn_geometry() -> dict:
    """Rendered vertices by fault id: every mesh of the seam faults on the run,
    and the edge of every visible rust patch with its islands."""
    data = _geometry_file()
    out = {fid: [tuple(p) for p in pts] for fid, pts in data.get("faults", {}).items()}
    for fid, rings in data.get("outlines", {}).items():
        out.setdefault(fid, [tuple(p) for ring in rings for p in ring])
    return out


@lru_cache(maxsize=1)
def drawn_outlines() -> dict:
    """The rendered edge of each visible rust patch and of each island, as
    closed rings of (theta, z)."""
    return {fid: [tuple((p[0], p[1]) for p in ring) for ring in rings]
            for fid, rings in _geometry_file().get("outlines", {}).items()}


def reload_geometry() -> None:
    """Forget the cached geometry, after the capture has written a new file."""
    for cached in (_geometry_file, drawn_geometry, drawn_outlines):
        cached.cache_clear()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def _inside(x: float, y: float, ring) -> bool:
    """Even-odd test of a point against a closed ring of (x, y)."""
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


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
    rings = drawn_outlines().get(d["id"])
    if rings:                                 # a rust patch, as it was rendered
        prepared = []
        for ring in rings:
            pts = [(_wrap(t - theta) * R, y) for t, y in ring]
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            prepared.append((min(xs), max(xs), min(ys), max(ys), pts))

        def drawn(t, y):
            x = _wrap(t - theta) * R
            return any(x0 <= x <= x1 and y0 <= y <= y1 and _inside(x, y, pts)
                       for x0, x1, y0, y1, pts in prepared)

        return drawn
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
def frame_samples(centre_theta: float, centre_z: float, inset: float):
    """Where a regular grid of points across the picture lands on the wall.

    Each sample is a ray from the lens through one point of the frame, cast
    onto the shell at the given depth. Sampling the picture rather than the
    wall means the grid is the frame by construction, wherever the lens is.
    """
    origin, right, up, back = camera_pose(centre_theta, centre_z)
    half_x = SENSOR_MM / 2 / LENS_MM
    half_y = half_x * RESOLUTION[1] / RESOLUTION[0]
    r = RADIUS - inset
    cols, rows = FRAME_SAMPLES
    out = []
    for j in range(rows):
        sy = (2 * (j + 0.5) / rows - 1) * half_y
        for i in range(cols):
            sx = (2 * (i + 0.5) / cols - 1) * half_x
            d = tuple(-back[k] + right[k] * sx + up[k] * sy for k in range(3))
            a = d[0] * d[0] + d[1] * d[1]
            b = 2 * (origin[0] * d[0] + origin[1] * d[1])
            c = origin[0] * origin[0] + origin[1] * origin[1] - r * r
            t = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
            hit = tuple(origin[k] + t * d[k] for k in range(3))
            out.append((math.atan2(hit[1], hit[0]), hit[2]))
    return tuple(out)


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
        shown = frame_samples(centre_theta, centre_z, round(depth, 6))
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
