"""Build the inside of a storage tank, its weld seams, its faults, and the crawler.

Run inside Blender:

    python scripts/blender_send.py tank/sim/blender_tank.py

Why a tank. A person enters a tank to look at its welds and its wall, and that
entry is what kills people: US confined spaces take about a hundred lives a
year, a quarter of them during inspection and cleaning, and most of the dead
are the second person who went in after the first. Emptying, cleaning and
gas-freeing a tank so that entry is legal is what makes an internal inspection
take weeks rather than days. A crawler that goes in instead has to do three
things a camera alone cannot: find the seams, judge them, and decide where the
ultrasonic probe is worth putting, because a thirty metre wall cannot be probed
everywhere.

What this file makes, and why each piece is here.

  the shell       six courses, thick at the bottom and thin at the top, the way
                  a tank is built, so wall thickness is a real function of
                  height rather than one number. Plate is rolled and shot
                  blasted, not polished, and it carries a stain line where the
                  product used to stand, which is where corrosion concentrates.
  the seams       a circumferential seam between every pair of courses, a
                  vertical seam at every plate join, staggered course to course
                  as a real tank staggers them, and the chime weld where the
                  shell meets the floor, which is the one that fails first.
  the beads       a weld is not a smooth tube. Near the crawler every bead is
                  built as the overlapping pools a welder actually leaves, with
                  a heat-tinted band either side; further away, where nothing
                  is judging it, a plain bead stands in. Detail costs polygons
                  and only the metal being looked at earns them.
  the faults      injected on purpose, each recorded before anything looks at
                  it, so every verdict can be marked against an answer written
                  in advance. Some are visible to a camera and some are not:
                  general wall loss shows nothing on the surface, and that is
                  the whole reason the probe has to be aimed rather than swept.
  the crawler     magnetic wheels, a sensor head on a boom ahead of the body so
                  the body is not in its own shot, two lamps set wide so a bead
                  throws a shadow, an ultrasonic probe in a sprung holder, and
                  the tether hanging down the wall behind it.

Ground truth goes to tank/data/tank_scene.json in metres, with the thinning
patches as parameters rather than a grid, so wall thickness can be asked for at
any point on the wall rather than at the points someone happened to sample.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# The shapes, surfaces and framing are shared with the bench rig, so both
# robots are built from the same primitives and a difference between them is a
# difference in the machine rather than in how it was modelled.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import meshkit  # noqa: E402

importlib.reload(meshkit)
from meshkit import (  # noqa: E402
    _principled,
    bead_profile,
    bead_surface,
    box,
    camera,
    chamfered,
    disc,
    ellipsoid,
    emissive,
    look_at,
    make,
    merge,
    plain,
    ring,
    shift,
    steel_material,
    sweep,
    tube,
)

# --------------------------------------------------------------------- shape

RADIUS = 10.0            # metres, a 20 m diameter tank
COURSE_H = 2.0           # each ring of plates
COURSES = 6              # 12 m tall
PLATES_PER_COURSE = 8    # plate joins, so eight vertical seams a course
SHELL_SEGMENTS = 256     # facets around: 245 mm of arc each, and it reads round

# Plate thickness by course, bottom first. A tank carries its own head, so the
# bottom course is the thick one and the top course is the thin one.
COURSE_MM = (12.0, 11.0, 10.0, 9.0, 8.0, 6.0)
FLOOR_MM = 8.0
PRODUCT_LINE_M = 7.4     # where the liquid used to stand, and where it stains

BEAD_R = 0.008           # a weld bead stands 8 mm proud of the plate
RIPPLE_PITCH = 0.009     # the spacing of the pools a welder leaves behind
DETAIL_ARC_M = 4.0       # how much seam either side of the crawler gets ripples
SEED = 7

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "tank" / "data" / "tank_scene.json"

# What counts as a fault worth reporting, so a brief can cite a limit rather
# than an opinion. It lives with the thing being made and is carried into the
# ground truth, so a scorer and a report read the same number.
ACCEPTANCE = {
    "undercut": {"limit_mm": 0.8, "measure": "depth below the plate surface"},
    "porosity": {"limit_count": 4, "measure": "pores in any 100 mm of bead"},
    "crack": {"limit_mm": 0.0, "measure": "any crack is rejectable"},
    "lack_of_fusion": {"limit_mm": 0.0, "measure": "any break in the bead"},
    "thinning": {"limit_fraction": 0.60,
                 "measure": "remaining wall as a fraction of the original"},
}

def shell_material():
    """Plate, plus the tide mark the product left behind.

    Every tank that has held a liquid carries a line. Above it the plate is dry
    and lighter; below it sits the staining, and with it most of the corrosion.
    It is the one detail that makes an interior read as a tank rather than a
    pipe, and it gives the crawler a landmark that is not a weld.
    """
    mat = steel_material("tank_plate", (0.185, 0.195, 0.215), 0.58, 0.85, 26.0,
                         0.11, fine=520.0)
    nt = mat.node_tree
    bsdf = _principled(mat)
    ramp_out = bsdf.inputs["Base Color"].links[0].from_socket
    coords = nt.nodes["Texture Coordinate"].outputs["Object"]

    axis = nt.nodes.new("ShaderNodeSeparateXYZ")
    axis.location = (-850, 460)
    nt.links.new(coords, axis.inputs["Vector"])

    wobble = nt.nodes.new("ShaderNodeTexNoise")
    wobble.location = (-850, 620)
    wobble.inputs["Scale"].default_value = 1.4
    nt.links.new(coords, wobble.inputs["Vector"])

    # The line is not ruled: it wanders a few centimetres as the liquid moved,
    # so the height it is tested against has to wander with it.
    height = nt.nodes.new("ShaderNodeMath")
    height.location = (-650, 520)
    height.operation = "SUBTRACT"
    nt.links.new(axis.outputs["Z"], height.inputs[0])
    nt.links.new(wobble.outputs["Fac"], height.inputs[1])

    band = nt.nodes.new("ShaderNodeMapRange")
    band.location = (-470, 520)
    band.inputs["From Min"].default_value = PRODUCT_LINE_M - 1.4
    band.inputs["From Max"].default_value = PRODUCT_LINE_M - 0.7
    band.clamp = True
    nt.links.new(height.outputs["Value"], band.inputs["Value"])

    stain = nt.nodes.new("ShaderNodeMixRGB")
    stain.location = (-260, 300)
    stain.blend_type = "MIX"
    stain.inputs["Color2"].default_value = (0.115, 0.098, 0.082, 1.0)
    nt.links.new(band.outputs["Result"], stain.inputs["Fac"])
    nt.links.new(ramp_out, stain.inputs["Color1"])
    nt.links.new(stain.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def build_materials() -> dict:
    return {
        "plate": shell_material(),
        "bead": steel_material("weld_bead", (0.215, 0.200, 0.170), 0.62, 0.62,
                               160.0, 0.22, fine=900.0),
        "haz": plain("heat_affected_zone", (0.150, 0.130, 0.100), 0.62, 0.60),
        "dark": plain("defect_dark", (0.018, 0.018, 0.022), 0.10, 0.88),
        "rust": steel_material("corrosion", (0.26, 0.105, 0.035), 0.92, 0.10, 70.0, 0.90),
        "body": plain("crawler_body", (0.46, 0.21, 0.035), 0.30, 0.52),
        "frame": plain("crawler_frame", (0.10, 0.11, 0.125), 0.80, 0.42),
        "tyre": plain("crawler_tyre", (0.045, 0.045, 0.05), 0.05, 0.75),
        "hub": plain("crawler_hub", (0.62, 0.64, 0.68), 0.95, 0.28),
        "glass": plain("lens_glass", (0.04, 0.06, 0.09), 0.15, 0.08),
        "lamp_face": emissive("lamp_face", (1.0, 0.96, 0.88), 14.0),
        "beacon": emissive("beacon", (0.15, 0.85, 0.35), 2.5),
        "tether": plain("tether", (0.06, 0.06, 0.07), 0.05, 0.85),
        "couplant": plain("couplant_hose", (0.10, 0.28, 0.52), 0.05, 0.60),
    }


def wall_point(theta, z, inset=0.0):
    r = RADIUS - inset
    return (r * math.cos(theta), r * math.sin(theta), z)


def wall_frame(theta):
    """Tangent, up and outward at a point on the wall, as world vectors."""
    return (Vector((-math.sin(theta), math.cos(theta), 0.0)),
            Vector((0.0, 0.0, 1.0)),
            Vector((math.cos(theta), math.sin(theta), 0.0)))


# ------------------------------------------------------------------ the beads


def smooth_bead_arc(z, arc, minor_r=BEAD_R):
    span = arc[1] - arc[0]
    segments = max(8, int(SHELL_SEGMENTS * span / math.tau))
    return ring(RADIUS - minor_r, minor_r, (0, 0, z), segments, 8, arc)


def rippled_bead_arc(z, arc, rng, minor_r=BEAD_R):
    """A circumferential seam, swept the long way round the tank."""
    span = arc[1] - arc[0]
    arc_len = span * RADIUS

    def place(f, across, out):
        return wall_point(arc[0] + span * f, z + across, inset=out)

    return bead_surface(place, arc_len, bead_profile(rng, arc_len, RIPPLE_PITCH),
                        minor_r, RIPPLE_PITCH)


def rippled_bead_column(theta, z0, z1, rng, minor_r=BEAD_R):
    """A vertical plate join, swept up the wall."""
    arc_len = z1 - z0

    def place(f, across, out):
        return wall_point(theta + across / RADIUS, z0 + arc_len * f, inset=out)

    return bead_surface(place, arc_len, bead_profile(rng, arc_len, RIPPLE_PITCH),
                        minor_r, RIPPLE_PITCH)


def haz_arc(z, arc, half_width=0.015, inset=0.0012):
    """The discoloured band either side of a weld, where the arc heated the plate."""
    span = arc[1] - arc[0]
    segments = max(8, int(SHELL_SEGMENTS * span / math.tau))
    parts = []
    for side in (-1, 1):
        lo = z + side * half_width * 0.38
        hi = z + side * half_width
        verts, faces = [], []
        for i in range(segments + 1):
            t = arc[0] + span * i / segments
            verts.append(wall_point(t, min(lo, hi), inset))
            verts.append(wall_point(t, max(lo, hi), inset))
        for i in range(segments):
            faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
        parts.append((verts, faces))
    return merge(parts)


# ------------------------------------------------------------------- the tank


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights,
                  bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)


def collection(name):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def build_shell(mats, coll):
    height = COURSE_H * COURSES
    # Divided up the height as well as around, so the shading gradient over a
    # twelve metre wall is smooth rather than banded.
    verts, faces = [], []
    rows = 24
    for r in range(rows + 1):
        z = height * r / rows
        for i in range(SHELL_SEGMENTS):
            t = math.tau * i / SHELL_SEGMENTS
            verts.append((RADIUS * math.cos(t), RADIUS * math.sin(t), z))
    for r in range(rows):
        for i in range(SHELL_SEGMENTS):
            k = (i + 1) % SHELL_SEGMENTS
            faces.append((r * SHELL_SEGMENTS + i, r * SHELL_SEGMENTS + k,
                          (r + 1) * SHELL_SEGMENTS + k, (r + 1) * SHELL_SEGMENTS + i))
    make("Tank_Shell", verts, faces, mats["plate"], collection=coll)

    make("Tank_Floor", *disc(RADIUS, 0.0), mat=mats["plate"], smooth=False,
         collection=coll)

    # Floor plates are lapped, so the floor carries its own welds in strips.
    laps = []
    for k in range(-4, 5):
        y = k * 2.1
        half = math.sqrt(max(0.0, RADIUS ** 2 - y ** 2))
        if half < 0.5:
            continue
        laps.append(box((half * 2, 0.022, 0.013), (0.0, y, 0.006)))
    make("Seam_Floor_Laps", *merge(laps), mat=mats["bead"], smooth=False,
         collection=coll)

    roof_v = [(0.0, 0.0, height + 1.4)]
    for i in range(SHELL_SEGMENTS):
        t = math.tau * i / SHELL_SEGMENTS
        roof_v.append((RADIUS * math.cos(t), RADIUS * math.sin(t), height))
    roof_f = [(0, i + 1, (i + 1) % SHELL_SEGMENTS + 1) for i in range(SHELL_SEGMENTS)]
    make("Tank_Roof", roof_v, roof_f, mats["plate"], collection=coll)

    # Rafters under the roof. Nobody inspects them here, but a cone roof with
    # nothing holding it up is the detail that makes a render look like a cone.
    beams = []
    for k in range(16):
        t = math.tau * k / 16
        length = RADIUS - 0.3
        verts, faces = box((length, 0.10, 0.24), (length / 2, 0.0, 0.0))
        c, s = math.cos(t), math.sin(t)
        beams.append(([(x * c - y * s, x * s + y * c,
                        z + height - 0.34 + (x / length) * 0.58)
                       for x, y, z in verts], faces))
    make("Roof_Rafters", *merge(beams), mat=mats["frame"], smooth=False, collection=coll)


def build_fittings(mats, coll):
    """The manway the crawler goes in by, and the only daylight in the tank."""
    theta, z = 0.0, 1.15
    rim = make("Manway_Rim", *ring(0.38, 0.055, (0, 0, 0), 48, 10), mat=mats["frame"],
               collection=coll)
    rim.location = wall_point(theta, z, inset=0.05)
    rim.rotation_euler = (math.pi / 2, 0.0, theta)

    bolts = []
    for k in range(12):
        a = math.tau * k / 12
        bolts.append(ellipsoid((0.38 * math.cos(a), 0.38 * math.sin(a), 0.0),
                               (0.024, 0.024, 0.024), segments=8))
    nuts = make("Manway_Bolts", *merge(bolts), mat=mats["hub"], collection=coll)
    nuts.location = wall_point(theta, z, inset=0.095)
    nuts.rotation_euler = (math.pi / 2, 0.0, theta)

    light = bpy.data.lights.new("Manway_Daylight", type="AREA")
    light.energy = 170.0
    light.size = 0.62
    obj = bpy.data.objects.new("Manway_Daylight", light)
    obj.matrix_world = look_at(wall_point(theta, z, inset=-0.25),
                               wall_point(theta, z, inset=1.2))
    coll.objects.link(obj)

    # A shell nozzle higher up, for scale and for something that is not a weld.
    noz = make("Shell_Nozzle", *tube(0.15, 0.36, 22), mat=mats["frame"], collection=coll)
    noz.location = wall_point(math.radians(200), 5.6, inset=0.16)
    noz.rotation_euler = (math.pi / 2, 0.0, math.radians(200))


def seam_table() -> list[dict]:
    """Every weld the crawler could be sent to, before any of it is built."""
    seams = [{"id": "CHIME", "kind": "chime", "z_m": round(BEAD_R, 4), "course": 0,
              "length_m": round(math.tau * RADIUS, 3)}]
    for c in range(1, COURSES):
        seams.append({"id": f"C{c}", "kind": "circumferential", "z_m": c * COURSE_H,
                      "course": c, "length_m": round(math.tau * RADIUS, 3)})
    step = math.tau / PLATES_PER_COURSE
    for c in range(COURSES):
        for p in range(PLATES_PER_COURSE):
            seams.append({"id": f"V{c}_{p}", "kind": "vertical", "course": c,
                          "theta_rad": round(p * step + (step / 2 if c % 2 else 0.0), 5),
                          "z_from_m": c * COURSE_H, "z_to_m": (c + 1) * COURSE_H,
                          "length_m": COURSE_H})
    return seams


def build_seams(mats, coll, seams, rng, detail_theta, detail_z) -> int:
    """Beads, with ripples only where something is going to look at them.

    Four hundred and seventy metres of weld modelled pool by pool is millions of
    polygons for metal no camera reaches. The window around the crawler gets the
    real thing; the rest gets a bead that reads correctly at a distance.
    """
    half = DETAIL_ARC_M / RADIUS / 2
    detailed = 0
    for seam in seams:
        if seam["kind"] in ("chime", "circumferential"):
            z = seam["z_m"]
            if abs(z - detail_z) < COURSE_H * 0.6:
                lo, hi = detail_theta - half, detail_theta + half
                make(f"Seam_{seam['id']}",
                     *merge([smooth_bead_arc(z, (hi, lo + math.tau)),
                             rippled_bead_arc(z, (lo, hi), rng)]),
                     mat=mats["bead"], collection=coll)
                make(f"HAZ_{seam['id']}", *haz_arc(z, (lo, hi)), mat=mats["haz"],
                     smooth=False, collection=coll)
                detailed += 1
            else:
                make(f"Seam_{seam['id']}", *smooth_bead_arc(z, (0.0, math.tau)),
                     mat=mats["bead"], collection=coll)
        else:
            theta = seam["theta_rad"]
            gap = abs(((theta - detail_theta + math.pi) % math.tau) - math.pi)
            if gap < half and abs(seam["z_from_m"] - detail_z) < COURSE_H:
                make(f"Seam_{seam['id']}",
                     *rippled_bead_column(theta, seam["z_from_m"], seam["z_to_m"], rng),
                     mat=mats["bead"], collection=coll)
                detailed += 1
            else:
                obj = make(f"Seam_{seam['id']}", *tube(BEAD_R, COURSE_H, 12),
                           mat=mats["bead"], collection=coll)
                obj.location = wall_point(theta, (seam["z_from_m"] + seam["z_to_m"]) / 2,
                                          inset=BEAD_R)
    return detailed


# ----------------------------------------------------------------- the faults


def plan_defects(seams, rng) -> list[dict]:
    """Decide the faults before any of them is drawn, and write them down."""
    circ = [s for s in seams if s["kind"] in ("circumferential", "chime")]
    vert = [s for s in seams if s["kind"] == "vertical"]
    out: list[dict] = []

    def add(kind, seam, theta, z, **extra):
        item = {"id": f"D{len(out) + 1:03d}", "kind": kind,
                "seam": seam["id"] if seam else None,
                "theta_rad": round(theta, 5), "z_m": round(z, 4),
                "centre_m": [round(v, 4) for v in wall_point(theta, z, BEAD_R)],
                "acceptance": ACCEPTANCE[kind]}
        item.update(extra)
        out.append(item)

    for _ in range(4):
        seam = rng.choice(circ)
        theta = rng.uniform(0, math.tau)
        length = rng.uniform(0.18, 0.55)
        depth = rng.uniform(0.5, 2.3)
        add("undercut", seam, theta + length / RADIUS / 2, seam["z_m"],
            length_mm=round(length * 1000, 1), depth_mm=round(depth, 2),
            arc_from=round(theta, 6), arc_to=round(theta + length / RADIUS, 6),
            rejectable=depth > ACCEPTANCE["undercut"]["limit_mm"], visible=True)

    for _ in range(4):
        seam = rng.choice(circ + vert)
        if seam["kind"] == "vertical":
            theta = seam["theta_rad"]
            z = rng.uniform(seam["z_from_m"] + 0.25, seam["z_to_m"] - 0.25)
        else:
            theta, z = rng.uniform(0, math.tau), seam["z_m"]
        count = rng.randint(5, 11)
        add("porosity", seam, theta, z, count=count,
            rejectable=count > ACCEPTANCE["porosity"]["limit_count"], visible=True)

    for _ in range(3):
        seam = rng.choice(circ)
        add("crack", seam, rng.uniform(0, math.tau), seam["z_m"],
            length_mm=round(rng.uniform(22, 85), 1), rejectable=True, visible=True)

    for _ in range(3):
        seam = rng.choice(vert)
        z = rng.uniform(seam["z_from_m"] + 0.35, seam["z_to_m"] - 0.35)
        add("lack_of_fusion", seam, seam["theta_rad"], z,
            length_mm=round(rng.uniform(55, 150), 1), rejectable=True, visible=True)

    for index in range(6):
        visible = index < 3
        course = rng.randint(0, COURSES - 1)
        z = rng.uniform(course * COURSE_H + 0.35, (course + 1) * COURSE_H - 0.35)
        nominal = COURSE_MM[course]
        loss = rng.uniform(1.5, 4.5)
        left = (nominal - loss) / nominal
        add("thinning", None, rng.uniform(0, math.tau), z,
            patch_radius_m=round(rng.uniform(0.25, 0.7), 3),
            loss_mm=round(loss, 2), nominal_mm=nominal,
            remaining_mm=round(nominal - loss, 2), remaining_fraction=round(left, 3),
            rejectable=left < ACCEPTANCE["thinning"]["limit_fraction"],
            falloff="cosine", visible=visible)
    return out


def build_defects(mats, coll, defects, rng):
    for d in defects:
        theta, z, kind = d["theta_rad"], d["z_m"], d["kind"]
        tangent, up, outward = wall_frame(theta)

        if kind == "undercut":
            # A groove melted into the plate at the toe of the weld, so it runs
            # beside the bead and never through it.
            lo, hi = d["arc_from"], d["arc_to"]
            depth = d["depth_mm"] / 1000
            segs = max(6, int((hi - lo) * RADIUS / 0.01))
            verts, faces = [], []
            for i in range(segs + 1):
                t = lo + (hi - lo) * i / segs
                edge = z + BEAD_R * 1.55
                verts.append(wall_point(t, edge, inset=-depth * 0.4))
                verts.append(wall_point(t, edge + 0.0048, inset=depth * 1.6))
            for i in range(segs):
                faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
            make(f"Defect_{d['id']}_undercut", verts, faces, mats["dark"],
                 collection=coll)

        elif kind == "porosity":
            pores = []
            for _ in range(d["count"]):
                r = rng.uniform(0.0011, 0.0029)
                centre = Vector(wall_point(theta + rng.uniform(-0.05, 0.05) / RADIUS,
                                           z + rng.uniform(-0.004, 0.004),
                                           inset=BEAD_R * 0.35))
                pores.append(ellipsoid(centre, (r, r, r * 0.8),
                                       frame=(tangent, up, outward), segments=8))
            make(f"Defect_{d['id']}_porosity", *merge(pores), mat=mats["dark"],
                 collection=coll)

        elif kind == "crack":
            length = d["length_mm"] / 1000
            verts, faces, steps = [], [], 12
            for i in range(steps + 1):
                # A crack wanders. A ruled line reads as a scratch.
                f = i / steps
                off = math.sin(f * 5.0) * 0.0016 + rng.uniform(-0.0004, 0.0004)
                zz = z - length / 2 + length * f
                verts.append(wall_point(theta + off / RADIUS, zz, inset=BEAD_R * 0.30))
                verts.append(wall_point(theta + (off + 0.0009) / RADIUS, zz,
                                        inset=BEAD_R * 0.30))
            for i in range(steps):
                faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
            make(f"Defect_{d['id']}_crack", verts, faces, mats["dark"], smooth=False,
                 collection=coll)

        elif kind == "lack_of_fusion":
            gap = d["length_mm"] / 1000
            verts, faces = [], []
            for i in range(9):
                zz = z - gap / 2 + gap * i / 8
                verts.append(wall_point(theta - 0.0024 / RADIUS, zz, inset=-0.0016))
                verts.append(wall_point(theta + 0.0024 / RADIUS, zz, inset=-0.0016))
            for i in range(8):
                faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
            make(f"Defect_{d['id']}_lack_of_fusion", verts, faces, mats["dark"],
                 smooth=False, collection=coll)

        elif kind == "thinning" and d["visible"]:
            # Wall loss a camera can see: a rusted patch, pitted through.
            patch = d["patch_radius_m"]
            segs = 40
            phase = rng.uniform(0, math.tau)

            def reach(a, phase=phase):
                """How far the rust runs at this bearing, so the edge is ragged."""
                return (0.70 + 0.20 * math.sin(3 * a + phase)
                        + 0.10 * math.sin(7 * a + phase * 1.7))

            skin, faces = [], []
            for i in range(segs + 1):
                for j in range(segs + 1):
                    a = math.tau * i / segs
                    rr = patch * reach(a) * (j / segs)
                    skin.append(wall_point(theta + math.cos(a) * rr / RADIUS,
                                           z + math.sin(a) * rr, inset=0.0009))
            for i in range(segs):
                for j in range(segs):
                    faces.append((i * (segs + 1) + j, i * (segs + 1) + j + 1,
                                  (i + 1) * (segs + 1) + j + 1,
                                  (i + 1) * (segs + 1) + j))
            make(f"Defect_{d['id']}_rust", skin, faces, mats["rust"], collection=coll)
            # Outliers: rust spreads in islands before it joins up.
            freckles = []
            for _ in range(rng.randint(5, 10)):
                a = rng.uniform(0, math.tau)
                rr = patch * rng.uniform(1.15, 1.9)
                small = patch * rng.uniform(0.07, 0.18)
                ring_v, ring_f = [], []
                for i in range(17):
                    b = math.tau * i / 16
                    for j in range(5):
                        q = small * (0.65 + 0.35 * math.sin(3 * b)) * (j / 4)
                        ring_v.append(wall_point(
                            theta + (math.cos(a) * rr + math.cos(b) * q) / RADIUS,
                            z + math.sin(a) * rr + math.sin(b) * q, inset=0.0008))
                for i in range(16):
                    for j in range(4):
                        base = i * 5 + j
                        ring_f.append((base, base + 1, base + 6, base + 5))
                freckles.append((ring_v, ring_f))
            make(f"Defect_{d['id']}_freckles", *merge(freckles), mat=mats["rust"],
                 collection=coll)

            pits = []
            for _ in range(rng.randint(40, 75)):
                a = rng.uniform(0, math.tau)
                rr = patch * reach(a) * math.sqrt(rng.uniform(0, 1))
                r = rng.uniform(0.004, 0.014)
                centre = Vector(wall_point(theta + math.cos(a) * rr / RADIUS,
                                           z + math.sin(a) * rr, inset=0.002))
                pits.append(ellipsoid(centre, (r, r, r * 0.55),
                                      frame=(tangent, up, outward), segments=8))
            make(f"Defect_{d['id']}_pits", *merge(pits), mat=mats["dark"],
                 collection=coll)


# ---------------------------------------------------------------- the crawler


def build_crawler(mats, coll, theta, z):
    """A magnetic crawler: wheels, a sensor head on a boom, a probe, a tether.

    Built in the robot's own frame and carried onto the wall by one matrix, so
    driving it along a seam later is one line rather than twenty. In that frame
    the plate is Z = 0, travel is +X, and +Y is down the wall, which is why the
    tether hangs off in +Y.
    """
    rig = bpy.data.objects.new("Crawler", None)
    rig.empty_display_size = 0.12
    coll.objects.link(rig)

    def part(name, verts, faces, mat, smooth=True):
        obj = make(name, verts, faces, mat, smooth=smooth, collection=coll)
        obj.parent = rig
        return obj

    part("Crawler_Deck", *box((0.50, 0.38, 0.028), (0.0, 0.0, 0.086)),
         mat=mats["frame"], smooth=False)
    part("Crawler_Body", *chamfered((0.34, 0.27, 0.125), (-0.01, 0.0, 0.163), 0.022),
         mat=mats["body"], smooth=False)
    for sy in (-0.145, 0.145):
        part(f"Crawler_Rib_{sy:+.3f}", *box((0.30, 0.018, 0.075), (-0.01, sy, 0.150)),
             mat=mats["frame"], smooth=False)

    # Magnetic wheels: a dark tyre on a bright hub, so they read as wheels in a
    # dark tank rather than as four more black boxes.
    tyres, hubs = [], []
    spin = Matrix.Rotation(math.pi / 2, 4, "X")
    for sx in (-0.185, 0.185):
        for sy in (-0.195, 0.195):
            place = Matrix.Translation((sx, sy, 0.078)) @ spin
            for src, sink in ((tube(0.078, 0.052, 24, caps=True), tyres),
                              (tube(0.036, 0.058, 16, caps=True), hubs)):
                v, f = src
                sink.append(([tuple(place @ Vector(p)) for p in v], f))
    part("Crawler_Tyres", *merge(tyres), mat=mats["tyre"])
    part("Crawler_Hubs", *merge(hubs), mat=mats["hub"])

    # The sensor head rides ahead of the body on two rails, because a camera
    # mounted on the body photographs the body.
    for sy in (-0.115, 0.115):
        part(f"Crawler_Rail_{sy:+.3f}", *box((0.30, 0.022, 0.022), (0.32, sy, 0.175)),
             mat=mats["frame"], smooth=False)
    part("Crawler_Head", *chamfered((0.11, 0.46, 0.058), (0.46, 0.0, 0.196), 0.012),
         mat=mats["body"], smooth=False)
    part("Crawler_Cam_Housing", *shift(tube(0.037, 0.072, 20, caps=True),
                                       (0.46, 0.0, 0.168)), mat=mats["frame"])
    part("Crawler_Lens", *shift(tube(0.030, 0.008, 20, caps=True), (0.46, 0.0, 0.131)),
         mat=mats["glass"])

    for sy in (-0.20, 0.20):
        part(f"Crawler_Lamp_Pod_{sy:+.2f}",
             *shift(tube(0.030, 0.052, 16, caps=True), (0.46, sy, 0.150)),
             mat=mats["frame"])
        part(f"Crawler_Lamp_Face_{sy:+.2f}",
             *shift(tube(0.026, 0.006, 16, caps=True), (0.46, sy, 0.122)),
             mat=mats["lamp_face"])

    # The ultrasonic probe sits in a sprung holder under the deck, with its
    # couplant hose running back to the body.
    part("Crawler_Probe_Holder", *box((0.075, 0.075, 0.055), (0.10, 0.0, 0.088)),
         mat=mats["frame"], smooth=False)
    part("Crawler_UT_Probe", *shift(tube(0.026, 0.055, 18, caps=True), (0.10, 0.0, 0.030)),
         mat=mats["frame"])
    part("Crawler_Couplant_Hose",
         *sweep([(0.10, 0.03, 0.075), (0.06, 0.07, 0.115), (-0.02, 0.06, 0.150),
                 (-0.10, 0.02, 0.170)], 0.005, 8), mat=mats["couplant"])

    part("Crawler_Beacon_Post", *shift(tube(0.016, 0.040, 12, caps=True),
                                       (-0.14, 0.0, 0.245)), mat=mats["frame"])
    part("Crawler_Beacon", *ellipsoid((-0.14, 0.0, 0.268), (0.013, 0.013, 0.013),
                                      segments=12), mat=mats["beacon"])

    # +Y is down the wall, so the tether falls away in +Y and drifts back.
    part("Crawler_Tether",
         *sweep([(-0.25, 0.02, 0.145), (-0.30, 0.30, 0.075), (-0.33, 0.95, 0.052),
                 (-0.36, 1.9, 0.046), (-0.38, 3.2, 0.044), (-0.40, 4.8, 0.043)],
                0.009, 8), mat=mats["tether"])


    # 14 mm at 126 mm off the plate covers 324 mm of seam across the frame, so a
    # bead is a seventh of the height and a run of pores is a pattern rather
    # than a landscape. It sits just under the lens ring, not level with the
    # housing: at 175 mm it was inside its own sensor head, which a 100 mm near
    # clip hid until the clip was set to something a robot can use.
    cam = camera("Crawler_Cam", 14.0, coll)
    cam.parent = rig
    cam.matrix_basis = look_at((0.46, 0.0, 0.126), (0.46, 0.0, 0.0),
                               up_hint=(0.0, 1.0, 0.0))

    # Watts, at a tenth of a metre. The first build used 260 and returned a white
    # frame: a lamp this close needs single figures, and it has to sit wide so a
    # bead casts a shadow instead of glaring back down the lens.
    for sy in (-0.20, 0.20):
        lamp = bpy.data.lights.new(f"Crawler_Lamp_{sy:+.2f}", type="SPOT")
        lamp.energy = 2.6
        lamp.spot_size = math.radians(115)
        lamp.spot_blend = 0.55
        lamp.shadow_soft_size = 0.012
        obj = bpy.data.objects.new(lamp.name, lamp)
        obj.parent = rig
        coll.objects.link(obj)
        obj.matrix_basis = look_at((0.46, sy, 0.118), (0.46, sy * 0.1, 0.0),
                                   up_hint=(1.0, 0.0, 0.0))

    # The inspection lamps light a saucer under the head, which is what a bead
    # needs and no use at all for driving. A separate flood throws light down
    # the seam ahead, the way a crawler actually carries two kinds of light.
    flood = bpy.data.lights.new("Crawler_Nav_Flood", type="SPOT")
    flood.energy = 26.0
    flood.spot_size = math.radians(88)
    flood.spot_blend = 0.75
    flood.shadow_soft_size = 0.04
    nav = bpy.data.objects.new(flood.name, flood)
    nav.parent = rig
    coll.objects.link(nav)
    nav.matrix_basis = look_at((0.28, 0.0, 0.27), (2.20, 0.0, 0.0),
                               up_hint=(0.0, 1.0, 0.0))

    tangent, _, outward = wall_frame(theta)
    normal = -outward                      # into the tank, away from the plate
    across = normal.cross(tangent)         # down the wall
    p = Vector(wall_point(theta, z))
    rig.matrix_world = Matrix((
        (tangent.x, across.x, normal.x, p.x),
        (tangent.y, across.y, normal.y, p.y),
        (tangent.z, across.z, normal.z, p.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return rig, cam


# ------------------------------------------------------------------- the shot


def build_work_lights(coll, theta):
    """What a crew rigs before anyone goes near a tank.

    Left to its own lamps the robot lights a dinner plate and every wide shot
    is black, which proves nothing about the geometry and says nothing to
    anyone watching. Two floods on the floor, well off to the side, light the
    near wall and leave the far side dark, which is what the inside of a tank
    under inspection actually looks like.
    """
    for k, offset in enumerate((-0.42, 0.38)):
        lamp = bpy.data.lights.new(f"Work_Light_{k}", type="SPOT")
        lamp.energy = 900.0
        lamp.spot_size = math.radians(78)
        lamp.spot_blend = 0.85
        lamp.shadow_soft_size = 0.25
        obj = bpy.data.objects.new(lamp.name, lamp)
        eye = wall_point(theta + offset, 0.9, inset=4.2)
        obj.matrix_world = look_at(eye, wall_point(theta + offset * 0.25, 4.6, inset=0.1))
        coll.objects.link(obj)


def build_cameras(coll, rig):
    cam = camera("Overview_Cam", 30.0, coll)
    cam.matrix_world = rig.matrix_world @ look_at((-2.3, -1.25, 1.55), (2.6, 0.10, 0.0))

    hero = camera("Hero_Cam", 35.0, coll)
    hero.matrix_world = rig.matrix_world @ look_at((-3.8, -2.9, 5.2), (1.6, 0.2, 0.0))
    return cam, hero


def set_render(scene, camera):
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = 1600
    scene.render.resolution_y = 900
    scene.camera = camera
    for attr, value in (("taa_render_samples", 96), ("use_raytracing", True),
                        ("use_shadows", True)):
        # Which of these exists depends on the Blender build, and a missing one
        # is not worth failing a scene over.
        with contextlib.suppress(AttributeError, TypeError):
            setattr(scene.eevee, attr, value)
    # A lamp on the robot is a hard, close light and a plain transform clips it
    # to white. AgX keeps the highlight and leaves the bead readable.
    for transform in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    for node in world.node_tree.nodes:
        if node.type == "BACKGROUND":
            # Not quite black: a sealed tank has some bounce, and a pure black
            # render hides whether the geometry is right.
            node.inputs[0].default_value = (0.026, 0.029, 0.035, 1.0)
            node.inputs[1].default_value = 1.0


# -------------------------------------------------------------------- the run


def main() -> dict:
    rng = random.Random(SEED)
    clear_scene()
    mats = build_materials()
    tank = collection("Tank")
    faults = collection("Defects")
    robot = collection("Crawler")

    seams = seam_table()
    defects = plan_defects(seams, rng)

    # Park on a fault rather than at a tidy round number, so what the camera
    # sees is something the ground truth already has an answer for, and put the
    # ripple detail where the robot actually is.
    subject = next((d for d in defects
                    if d["kind"] in ("porosity", "undercut", "crack")
                    and d["seam"] and d["seam"].startswith("C")), defects[0])
    park_theta = subject["theta_rad"] - 0.46 / RADIUS
    park_z = subject["z_m"]

    build_shell(mats, tank)
    build_fittings(mats, tank)
    detailed = build_seams(mats, tank, seams, rng, park_theta, park_z)
    build_defects(mats, faults, defects, rng)
    rig, crawler_cam = build_crawler(mats, robot, park_theta, park_z)
    build_work_lights(tank, park_theta)
    overview, hero = build_cameras(robot, rig)
    set_render(bpy.context.scene, overview)

    ground_truth = {
        "units": "metres unless a key says mm",
        "seed": SEED,
        "tank": {"radius_m": RADIUS, "height_m": COURSE_H * COURSES,
                 "courses": COURSES, "course_height_m": COURSE_H,
                 "plates_per_course": PLATES_PER_COURSE,
                 "course_thickness_mm": list(COURSE_MM),
                 "floor_thickness_mm": FLOOR_MM,
                 "product_line_m": PRODUCT_LINE_M},
        "acceptance": ACCEPTANCE,
        "seams": seams,
        "defects": defects,
        "crawler": {"parked_at": {"theta_rad": round(park_theta, 5), "z_m": park_z},
                    "looking_at": subject["id"], "camera": "Crawler_Cam",
                    "probe": "Crawler_UT_Probe",
                    "camera_standoff_m": 0.126, "camera_lens_mm": 14.0},
        "detail_window": {"centre_theta_rad": round(park_theta, 5),
                          "arc_m": DETAIL_ARC_M, "ripple_pitch_m": RIPPLE_PITCH},
        "thickness_model": (
            "nominal is course_thickness_mm for the course a point sits in. Every "
            "thinning defect subtracts loss_mm * cos(pi/2 * d / patch_radius_m) for "
            "a point d metres from its centre, and nothing beyond that radius."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ground_truth, indent=1), encoding="utf-8")

    kinds: dict[str, int] = {}
    for d in defects:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    return {
        "objects": len(bpy.data.objects),
        "polygons": sum(len(o.data.polygons) for o in bpy.data.objects
                        if getattr(o, "data", None) and hasattr(o.data, "polygons")),
        "seams": len(seams),
        "seam_metres": round(sum(s["length_m"] for s in seams), 1),
        "seams_with_ripples": detailed,
        "defects": kinds,
        "rejectable": sum(1 for d in defects if d.get("rejectable")),
        "hidden_thinning": sum(1 for d in defects
                               if d["kind"] == "thinning" and not d["visible"]),
        "parked_on": {"defect": subject["id"], "kind": subject["kind"],
                      "seam": subject["seam"]},
        "ground_truth": str(OUT),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
