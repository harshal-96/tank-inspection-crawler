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

# The run the crawler is sent to inspect. Faults were scattered over 473 metres
# of weld before this existed, which meant a crawler could drive the whole of
# its detailed window and pass none of them: a dataset of clean frames and no
# way to score anything. An inspection is a run, and the faults found are the
# ones in it, so the run is chosen first and faults are planted along it.
SURVEY_SEAM = "C2"
SURVEY_CENTRE_RAD = 0.0
SURVEY_ARC_M = 4.0

BEAD_R = 0.008           # a weld bead stands 8 mm proud of the plate
RIPPLE_PITCH = 0.009     # the spacing of the pools a welder leaves behind
DETAIL_ARC_M = SURVEY_ARC_M + 1.2   # the run, plus enough that its edge is
                                    # never inside a frame. Matched exactly to
                                    # the run, the last stations looked past
                                    # the end of the modelled bead at a smooth
                                    # tube, and the inspector reported a break
                                    # in the weld. It was right: there was one.
# The running gear, named here because the detail pass hangs parts off these
# wheels and the animation has to know how big they are to roll them.
WHEEL_R = 0.078          # tyre radius, and therefore metres per radian
WHEEL_WIDTH = 0.052
WHEEL_Z = 0.078          # hub height above the plate
WHEEL_AT = ((-0.185, -0.195), (-0.185, 0.195),
            (0.185, -0.195), (0.185, 0.195))

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
        "haz": plain("heat_affected_zone", (0.128, 0.115, 0.094), 0.58, 0.62),
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
        "reservoir": plain("couplant_tank", (0.62, 0.66, 0.70), 0.02, 0.28),
    }


def survey_window():
    """The arc the crawler drives, as a pair of angles."""
    half = SURVEY_ARC_M / RADIUS / 2
    return SURVEY_CENTRE_RAD - half, SURVEY_CENTRE_RAD + half


def in_window(theta, margin=0.0):
    lo, hi = survey_window()
    wrapped = ((theta - SURVEY_CENTRE_RAD + math.pi) % math.tau) - math.pi
    return abs(wrapped) <= (hi - lo) / 2 + margin


def wall_point(theta, z, inset=0.0):
    r = RADIUS - inset
    return (r * math.cos(theta), r * math.sin(theta), z)


# How far each skin on the plate stands toward the camera, in metres. The
# order matters more than the numbers: nothing in this scene is subtracted, so
# a surface hides everything behind it, and a pit whose floor is behind the
# sheet it sits on shows as a ring rather than a hole.
HAZ_INSET = 0.0004        # heat tint, laid straight on the plate
RUST_INSET = 0.0009       # rust over the tint, because that is the order it happens
PIT_FLOOR = 0.00105       # the bottom of a pit, still clear of the rust
UNDERCUT_FLOOR = 0.0010   # the bottom of the groove, just clear of the rust


def crater(theta, z, r, depth, rim_inset, segments=14, phase=0.0):
    """A pit: a ragged rim at the surface and a floor set back behind it.

    Modelled as a small sphere at the wall a pit becomes a boulder sitting on
    the plate, because most of the sphere is in front of the surface it is
    supposed to be eating into. That is what the frames showed, and the
    inspector called them loose debris rather than corrosion, which is a fair
    description of a lump on sound metal. A pit has to be cut, not stuck on.
    """
    verts = [wall_point(theta, z, inset=rim_inset - depth)]
    for frac, dfrac in ((0.58, 0.78), (1.0, 0.0)):
        for i in range(segments):
            b = math.tau * i / segments
            rr = r * frac * (0.88 + 0.08 * math.sin(3 * b + phase)
                             + 0.04 * math.sin(5 * b + phase * 1.6))
            verts.append(wall_point(theta + math.cos(b) * rr / RADIUS,
                                    z + math.sin(b) * rr,
                                    inset=rim_inset - depth * dfrac))
    faces = [(0, 1 + i, 1 + (i + 1) % segments) for i in range(segments)]
    inner, outer = 1, 1 + segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((inner + i, inner + j, outer + j, outer + i))
    return verts, faces


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


def haz_arc(z, arc, half_width=0.0090, inset=HAZ_INSET):
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


def build_seams(mats, coll, seams, rng, detail_theta, detail_z, defects=()) -> int:
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
                # Where a fault says the bead stops, the bead stops. Drawn
                # through the gap it is a dark line painted on unbroken metal,
                # which is not what a break in a weld looks like.
                gaps = sorted((d["arc_from"], d["arc_to"]) for d in defects
                              if d["kind"] == "lack_of_fusion"
                              and d.get("seam") == seam["id"] and "arc_from" in d)
                spans, cursor = [], lo
                for g0, g1 in gaps:
                    if g0 > cursor:
                        spans.append((cursor, g0))
                    cursor = max(cursor, g1)
                spans.append((cursor, hi))
                parts = [smooth_bead_arc(z, (hi, lo + math.tau))]
                parts += [rippled_bead_arc(z, span, rng) for span in spans
                          if (span[1] - span[0]) * RADIUS > 0.02]
                make(f"Seam_{seam['id']}", *merge(parts), mat=mats["bead"],
                     collection=coll)
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
    """Decide the faults before any of them is drawn, and write them down.

    Four go on the run the crawler is sent to, spaced along it so consecutive
    frames do not all carry the same one, and two thinning patches sit beside
    it, one of them showing nothing on the surface. The rest are scattered over
    the other 469 metres, which is the tank the crawler has not reached yet.
    """
    circ = [s for s in seams if s["kind"] in ("circumferential", "chime")]
    vert = [s for s in seams if s["kind"] == "vertical"]
    run = next(s for s in seams if s["id"] == SURVEY_SEAM)
    lo, hi = survey_window()
    out: list[dict] = []

    def add(kind, seam, theta, z, **extra):
        item = {"id": f"D{len(out) + 1:03d}", "kind": kind,
                "seam": seam["id"] if seam else None,
                "theta_rad": round(theta, 5), "z_m": round(z, 4),
                "centre_m": [round(v, 4) for v in wall_point(theta, z, BEAD_R)],
                "on_survey_run": bool(seam and seam["id"] == SURVEY_SEAM
                                      and in_window(theta)),
                "acceptance": ACCEPTANCE[kind]}
        item.update(extra)
        out.append(item)

    def at(fraction):
        return lo + (hi - lo) * fraction

    # ---- on the run, at known places along it
    depth = 1.7
    length = 0.42
    add("undercut", run, at(0.17) + length / RADIUS / 2, run["z_m"],
        length_mm=round(length * 1000, 1), depth_mm=depth,
        arc_from=round(at(0.17), 6), arc_to=round(at(0.17) + length / RADIUS, 6),
        rejectable=depth > ACCEPTANCE["undercut"]["limit_mm"], visible=True)
    add("porosity", run, at(0.39), run["z_m"], count=7,
        rejectable=ACCEPTANCE["porosity"]["limit_count"] < 7, visible=True)
    add("crack", run, at(0.60), run["z_m"], length_mm=64.0, rejectable=True,
        visible=True)
    gap = 0.11
    add("lack_of_fusion", run, at(0.81) + gap / RADIUS / 2, run["z_m"],
        length_mm=round(gap * 1000, 1),
        arc_from=round(at(0.81), 6), arc_to=round(at(0.81) + gap / RADIUS, 6),
        rejectable=True, visible=True)

    # ---- beside the run: one wall loss a camera can see, one it cannot
    # Within reach of the probe arm, which is what makes them findable at all,
    # and the visible one close enough to the seam that the crawler's camera
    # actually passes over it. At 340 mm off the seam it was outside every
    # frame, so the run had no corrosion a camera could find and the contrast
    # the whole thing rests on was not in the dataset.
    #
    # The one a camera can see is within limits; the one it cannot see is not.
    # The patches straddle a course boundary, so the same loss is a different
    # fraction either side of it: 4.7 mm off the 11 mm course left it a hair
    # inside the limit and the run had nothing the probe had to catch.
    # One visible patch inside limits, and five hidden ones that are not.
    #
    # The count is deliberate. With a single hidden patch the run held four
    # rejectable probe points out of 168, and two planners compared on four
    # points produce an exact McNemar of 1.00, meaning no evidence either way.
    # A test set has to be able to separate the things being tested. Spread at
    # different heights so they fall across different parts of the probe grid
    # rather than stacking on the same rows.
    #
    # This is more severe loss than a tank of this age would really carry, and
    # it is marked in the ground truth as an evaluation scene so that nobody
    # reads it as a claim about real shells.
    for fraction, visible, loss, offset in (
            (0.28, True, 3.1, 0.075),
            (0.70, False, 5.6, -0.41),
            (0.13, False, 6.8, -0.30),
            (0.42, False, 6.2, 0.33),
            (0.57, False, 7.2, -0.37),
            (0.87, False, 6.5, 0.27)):
        z = run["z_m"] + offset
        nominal = COURSE_MM[min(COURSES - 1, int(z // COURSE_H))]
        left = (nominal - loss) / nominal
        add("thinning", None, at(fraction), z, patch_radius_m=0.38,
            loss_mm=loss, nominal_mm=nominal, remaining_mm=round(nominal - loss, 2),
            remaining_fraction=round(left, 3),
            rejectable=left < ACCEPTANCE["thinning"]["limit_fraction"],
            falloff="cosine", visible=visible)

    # ---- the rest of the tank, which the crawler has not been to
    def away(seam):
        for _ in range(40):
            theta = rng.uniform(0, math.tau)
            if seam["id"] != SURVEY_SEAM or not in_window(theta, 0.12):
                return theta
        return SURVEY_CENTRE_RAD + math.pi

    for _ in range(3):
        seam = rng.choice(circ)
        theta = away(seam)
        length = rng.uniform(0.18, 0.55)
        depth = rng.uniform(0.5, 2.3)
        add("undercut", seam, theta + length / RADIUS / 2, seam["z_m"],
            length_mm=round(length * 1000, 1), depth_mm=round(depth, 2),
            arc_from=round(theta, 6), arc_to=round(theta + length / RADIUS, 6),
            rejectable=depth > ACCEPTANCE["undercut"]["limit_mm"], visible=True)

    for _ in range(3):
        seam = rng.choice(circ + vert)
        if seam["kind"] == "vertical":
            theta = seam["theta_rad"]
            z = rng.uniform(seam["z_from_m"] + 0.25, seam["z_to_m"] - 0.25)
        else:
            theta, z = away(seam), seam["z_m"]
        count = rng.randint(5, 11)
        add("porosity", seam, theta, z, count=count,
            rejectable=count > ACCEPTANCE["porosity"]["limit_count"], visible=True)

    for _ in range(2):
        seam = rng.choice(circ)
        add("crack", seam, away(seam), seam["z_m"],
            length_mm=round(rng.uniform(22, 85), 1), rejectable=True, visible=True)

    for _ in range(3):
        seam = rng.choice(vert)
        z = rng.uniform(seam["z_from_m"] + 0.35, seam["z_to_m"] - 0.35)
        add("lack_of_fusion", seam, seam["theta_rad"], z,
            length_mm=round(rng.uniform(55, 150), 1), rejectable=True, visible=True)

    for index in range(4):
        visible = index < 2
        course = rng.randint(0, COURSES - 1)
        z = rng.uniform(course * COURSE_H + 0.35, (course + 1) * COURSE_H - 0.35)
        nominal = COURSE_MM[course]
        loss = rng.uniform(1.5, 4.5)
        left = (nominal - loss) / nominal
        # Off the run, or a patch nobody planted turns up in the frames and the
        # dataset says a clean length of seam had corrosion on it.
        theta = rng.uniform(0, math.tau)
        while in_window(theta, 0.15) and abs(z - COURSE_H * 2) < 1.0:
            theta = rng.uniform(0, math.tau)
        add("thinning", None, theta, z,
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
            width = 0.0046
            # Starts under the bead, not beside it. The edge of the bead is at
            # one bead radius from the seam centre, so beginning at 0.86 of one
            # puts the first row of the groove underneath the overhang, which
            # is where the shadow that gives an undercut away comes from.
            toe = z + BEAD_R * 0.86
            # Fine enough to carry a scallop per ripple, which needs several
            # sections inside each nine millimetres of weld.
            segs = max(48, int((hi - lo) * RADIUS / 0.0015))
            # Across the groove: rim, shoulder, floor, floor, shoulder, rim, as
            # a fraction of the width and a fraction of the depth above the
            # floor. Six rows rather than three, because a V of two quads has
            # no wall to catch the light on.
            # Under the bead, floor, floor, shoulder, lip. Not symmetric,
            # because the bead is on one side and bare plate on the other.
            cross = ((0.00, 0.06), (0.30, 0.00), (0.60, 0.00),
                     (0.82, 0.48), (1.00, 0.88))
            # The scallop. One period per weld ripple, so the groove pinches
            # where a crest reaches the plate and opens in the valley behind
            # it, and the whole thing stops looking ruled.
            # Twelve millimetres of fade at each end, not a seventh of the
            # length. On a long groove a proportional fade leaves tens of
            # millimetres at no depth at all, which the truth still counted as
            # an undercut and no camera could have found.
            fade = min(0.35, 0.012 / max(1e-6, (hi - lo) * RADIUS))

            def section(i):
                f = i / segs
                along = (hi - lo) * f * RADIUS
                beat = math.tau * along / (RIPPLE_PITCH * 3.5)
                # Never below about half width. Pinched to nothing at each
                # ripple the groove stopped being one groove and became a row
                # of dark blocks along the toe.
                swell = (0.78 + 0.16 * math.sin(beat)
                         + 0.06 * math.sin(beat * 2.7 + 1.1))
                drift = 0.0006 * math.sin(beat * 0.8) + 0.00025 * math.sin(along * 31.0)
                taper = min(1.0, f / fade, (1.0 - f) / fade)
                deep = (0.80 + 0.20 * math.sin(beat * 1.6 + 0.5)
                        + 0.10 * math.sin(along * 17.0))
                return (lo + (hi - lo) * f, toe + drift, width * swell,
                        taper * deep)

            verts, faces = [], []
            for i in range(segs + 1):
                t, at, wide, deep = section(i)
                for across, up in cross:
                    verts.append(wall_point(
                        t, at + wide * across,
                        inset=UNDERCUT_FLOOR + depth * up * deep))
            rows = len(cross)
            for i in range(segs):
                for j in range(rows - 1):
                    base = i * rows + j
                    faces.append((base, base + 1, base + rows + 1, base + rows))
            # Cut steel, flat shaded. Both of those matter. Smooth shading
            # averages the two walls into one even surface and the notch
            # disappears; painting the whole trough near-black replaces a lit
            # lip and a shadowed slot with a single dark smear, which beside a
            # weld is exactly what heat tint looks like, and got called that.
            make(f"Defect_{d['id']}_undercut", verts, faces, mats["plate"],
                 smooth=False, collection=coll)

            # The bottom of the slot, dark, a hair in front of the floor it
            # lies on. This is the part a camera sees as the undercut.
            fv, ff = [], []
            for i in range(segs + 1):
                t, at, wide, deep = section(i)
                for across in (0.24, 0.66):
                    fv.append(wall_point(t, at + wide * across,
                                         inset=UNDERCUT_FLOOR + 0.00006))
            for i in range(segs):
                ff.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
            make(f"Defect_{d['id']}_undercut_floor", fv, ff, mats["dark"],
                 smooth=False, collection=coll)

        elif kind == "porosity":
            # Cut into the crown of the bead. Set at a third of the bead
            # radius from the wall the pores sat five millimetres underneath
            # the surface of the weld, inside solid metal, and no camera saw
            # one: the frames carrying porosity showed a sound bead, and the
            # inspector named the only dark thing that was actually in them.
            pores = []
            for _ in range(d["count"]):
                r = rng.uniform(0.0011, 0.0029)
                pores.append(crater(theta + rng.uniform(-0.05, 0.05) / RADIUS,
                                    z + rng.uniform(-0.0022, 0.0022),
                                    r, r * 1.7, BEAD_R * 1.02,
                                    phase=rng.uniform(0, math.tau)))
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
            # The bead is simply absent, so what shows is the bare joint. On a
            # circumferential seam the gap runs round the tank; on a vertical
            # one it runs up it.
            verts, faces = [], []
            if "arc_from" in d:
                lo_a, hi_a = d["arc_from"], d["arc_to"]
                for i in range(13):
                    t = lo_a + (hi_a - lo_a) * i / 12
                    verts.append(wall_point(t, z - 0.0024, inset=-0.0016))
                    verts.append(wall_point(t, z + 0.0024, inset=-0.0016))
            else:
                gap = d["length_mm"] / 1000
                for i in range(9):
                    zz = z - gap / 2 + gap * i / 8
                    verts.append(wall_point(theta - 0.0024 / RADIUS, zz, inset=-0.0016))
                    verts.append(wall_point(theta + 0.0024 / RADIUS, zz, inset=-0.0016))
            for i in range(len(verts) // 2 - 1):
                faces.append((2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2))
            make(f"Defect_{d['id']}_lack_of_fusion", verts, faces, mats["dark"],
                 smooth=False, collection=coll)

        elif kind == "thinning" and d["visible"]:
            # Wall loss a camera can see: a rusted patch, pitted through.
            patch = d["patch_radius_m"]
            # Around the edge, and out from the middle. Forty steps around a
            # patch this size is a sixty millimetre straight line per step, and
            # the crawler works close enough to see every one of them: the
            # boundary came out as a drawn polygon. The outline is what gives a
            # patch away, so the detail goes there and not into the middle.
            around, out = 180, 12
            phase = rng.uniform(0, math.tau)

            def reach(a, phase=phase):
                """How far the rust runs at this bearing, so the edge is ragged."""
                return (0.70 + 0.20 * math.sin(3 * a + phase)
                        + 0.10 * math.sin(7 * a + phase * 1.7)
                        + 0.05 * math.sin(13 * a + phase * 2.3)
                        + 0.025 * math.sin(29 * a + phase * 0.6))

            skin, faces = [], []
            for i in range(around + 1):
                a = math.tau * i / around
                for j in range(out + 1):
                    rr = patch * reach(a) * (j / out)
                    skin.append(wall_point(theta + math.cos(a) * rr / RADIUS,
                                           z + math.sin(a) * rr, inset=RUST_INSET))
            for i in range(around):
                for j in range(out):
                    faces.append((i * (out + 1) + j, i * (out + 1) + j + 1,
                                  (i + 1) * (out + 1) + j + 1,
                                  (i + 1) * (out + 1) + j))
            make(f"Defect_{d['id']}_rust", skin, faces, mats["rust"], collection=coll)
            # Outliers: rust spreads in islands before it joins up. Small, and
            # pitted like the main patch, because a smooth orange shape at a
            # hand's distance reads as a sticker rather than as corrosion.
            def on_a_weld(y, clear):
                """Is this close enough to a seam to land on the bead itself?

                The patch is planted beside the seam and reaches further than
                its nominal radius, so pits scattered over it were landing on
                the weld crown. One of them showed in a frame as a single dark
                round pit in the face of the bead, which is the definition of
                porosity, and the inspector reported porosity on a sound weld.
                It was reading the picture correctly. Rust does not grow on a
                bead in the first place, so it is kept off one.
                """
                return min(abs(y - k * COURSE_H)
                           for k in range(COURSES + 1)) < clear

            freckles, islands, steps, rings = [], [], 24, 4
            for _ in range(rng.randint(14, 26)):
                a = rng.uniform(0, math.tau)
                rr = patch * rng.uniform(1.05, 1.85)
                small = patch * rng.uniform(0.03, 0.085)
                ph = rng.uniform(0, math.tau)
                if on_a_weld(z + math.sin(a) * rr, 0.030 + small):
                    continue
                islands.append((a, rr, small))
                ring_v, ring_f = [], []
                for i in range(steps + 1):
                    b = math.tau * i / steps
                    # Two harmonics at a tenth each, not one at a third. An
                    # island wants an edge that wanders, and the old one had
                    # three deep bays in it that read as a drawn shape.
                    edge = small * (0.82 + 0.12 * math.sin(3 * b + ph)
                                    + 0.06 * math.sin(7 * b + ph * 1.7))
                    for j in range(rings + 1):
                        q = edge * j / rings
                        ring_v.append(wall_point(
                            theta + (math.cos(a) * rr + math.cos(b) * q) / RADIUS,
                            z + math.sin(a) * rr + math.sin(b) * q,
                            inset=RUST_INSET))
                for i in range(steps):
                    for j in range(rings):
                        base = i * (rings + 1) + j
                        ring_f.append((base, base + 1,
                                       base + rings + 2, base + rings + 1))
                freckles.append((ring_v, ring_f))
            make(f"Defect_{d['id']}_freckles", *merge(freckles), mat=mats["rust"],
                 collection=coll)
            # Written back into the defect, because the ground truth is dumped
            # after this runs and a patch of rust is not only its middle. An
            # island a metre out is still rust on the wall, and a frame holding
            # one holds corrosion.
            d["rust_islands"] = [
                [round(theta + math.cos(ia) * irr / RADIUS, 6),
                 round(z + math.sin(ia) * irr, 4), round(ismall, 4)]
                for ia, irr, ismall in islands]

            # Pit count follows the area, not a fixed number. Forty pits over a
            # patch this size is a dozen in a close-up frame, and a smooth
            # orange field at a hand-s distance is not what rust looks like.
            # Small and many, because pitting is. The old ones ran to sixteen
            # millimetres of radius stuck two millimetres into the wall, so
            # each was a faceted lump the width of the weld standing proud of
            # the plate, and a frame of them looked like gravel tipped down the
            # tank rather than a corroded shell.
            pits = []
            for _ in range(int(6000 * patch * patch) + rng.randint(40, 90)):
                if rng.random() < 0.18 and islands:
                    # On an island, not merely out where the islands are. Pits
                    # were scattered to nearly twice the patch radius while the
                    # rust itself reached only just past it, so a frame could
                    # hold one dark round pit on clean grey plate with no rust
                    # anywhere in it. Beside a seam that is a pore in a weld,
                    # and it was reported as one, correctly.
                    ia, irr, ismall = islands[rng.randrange(len(islands))]
                    ang = rng.uniform(0, math.tau)
                    q = ismall * 0.68 * math.sqrt(rng.uniform(0, 1))
                    pit_t = theta + (math.cos(ia) * irr + math.cos(ang) * q) / RADIUS
                    pit_z = z + math.sin(ia) * irr + math.sin(ang) * q
                else:
                    a = rng.uniform(0, math.tau)
                    rr = patch * reach(a) * math.sqrt(rng.uniform(0, 1))
                    pit_t = theta + math.cos(a) * rr / RADIUS
                    pit_z = z + math.sin(a) * rr
                # Clear of the bead and of the heat-tinted band beside it.
                if on_a_weld(pit_z, 0.026):
                    continue
                # Nearly flat. Given a rim of its own height a pit becomes a
                # lump with a lit edge, because nothing here removes metal and
                # a dish standing proud of the plate is a dish standing proud
                # of the plate, lit on top. Dark, round and barely dished is
                # what reads as a pit, and the floor of it has to stay in front
                # of the rust it is eating into or the rust simply covers it.
                r = rng.uniform(0.002, 0.005)
                rim = PIT_FLOOR + min(0.0004, r * 0.12)
                pits.append(crater(pit_t, pit_z, r, rim - PIT_FLOOR, rim,
                                   segments=14, phase=rng.uniform(0, math.tau)))
            make(f"Defect_{d['id']}_pits", *merge(pits), mat=mats["dark"],
                 collection=coll)


# ---------------------------------------------------------------- the crawler


def crawler_matrix(theta, z) -> Matrix:
    """Where the crawler sits on the wall, as one matrix.

    Its own frame: +X the way it drives along the seam, +Y down the wall, +Z
    off the plate into the tank. Driving it is then one assignment, which is
    what the capture run needs.
    """
    tangent, _, outward = wall_frame(theta)
    normal = -outward                      # into the tank, away from the plate
    across = normal.cross(tangent)         # down the wall
    p = Vector(wall_point(theta, z))
    return Matrix((
        (tangent.x, across.x, normal.x, p.x),
        (tangent.y, across.y, normal.y, p.y),
        (tangent.z, across.z, normal.z, p.z),
        (0.0, 0.0, 0.0, 1.0),
    ))


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
    #
    # One object per wheel, each an empty at its own hub centre with the
    # geometry built around the origin. Merged into a single mesh they are
    # cheaper and they cannot turn, and a crawler that slides a metre with its
    # wheels locked is the first thing anybody notices.
    spin = Matrix.Rotation(math.pi / 2, 4, "X")   # axle along local Y
    for index, (sx, sy) in enumerate(WHEEL_AT):
        axle = bpy.data.objects.new(f"Crawler_Wheel_{index}", None)
        axle.empty_display_size = 0.04
        axle.rotation_mode = "XYZ"
        coll.objects.link(axle)
        axle.parent = rig
        axle.location = (sx, sy, WHEEL_Z)
        for suffix, src, mat in (
                ("Tyre", tube(WHEEL_R, WHEEL_WIDTH, 24, caps=True), mats["tyre"]),
                ("Hub", tube(0.036, 0.058, 16, caps=True), mats["hub"])):
            v, f = src
            obj = make(f"Crawler_Wheel_{index}_{suffix}",
                       [tuple(spin @ Vector(p)) for p in v], f, mat,
                       collection=coll)
            obj.parent = axle

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

    # The ultrasonic probe, and the water that makes it work.
    #
    # Ultrasound will not cross air, so the only thing that matters is the film
    # between the probe face and the steel. Everything here exists to put water
    # in that gap and keep it there while the wheel turns: a tank the robot
    # carries, a pump, a line down the side of the probe, and a nozzle that
    # delivers at the face rather than near it.
    #
    # Water and not gel. Gel will not pump through a small bore and it dries on
    # a vertical wall. Pumped systems and wheel probes both run water.
    part("Crawler_Probe_Holder", *box((0.075, 0.075, 0.055), (0.10, 0.0, 0.088)),
         mat=mats["frame"], smooth=False)
    # Dropped so the face all but touches the plate. It sat two and a half
    # millimetres clear, which is a probe hovering over the steel rather than
    # pressed into it, and it contradicted the spring above it. What is left
    # is the couplant film, and that is the only gap there should ever be.
    part("Crawler_UT_Probe",
         *shift(tube(0.026, 0.055, 18, caps=True), (0.10, 0.0, 0.0279)),
         mat=mats["frame"])

    # The spring. The face has to be pushed into the plate hard enough to hold
    # a film and give enough to ride over a weld toe and a plate step, and a
    # holder described as sprung with nothing sprung in it is a drawing of a
    # rigid probe that would skip.
    coil = []
    turns, steps = 6, 72
    for i in range(steps + 1):
        f = i / steps
        a = math.tau * turns * f
        coil.append((0.10 + 0.031 * math.cos(a), 0.031 * math.sin(a),
                     0.040 + 0.021 * f))
    part("Crawler_Probe_Spring", *sweep(coil, 0.0022, 6), mat=mats["hub"])

    # The tank it carries, and the pump. Both sit on the deck where a person
    # can reach them, because the water is the consumable and somebody has to
    # fill it between runs. How long a survey can be is set here, not by the
    # battery.
    # On the lid. The body shell is 125 deep and the tank was inside it, which
    # solids in a scene will happily allow and a fabricator will not. The lid
    # is also where it belongs: the water is the consumable, and whoever
    # refills it should not have to take the deck off to reach it.
    part("Crawler_Couplant_Tank",
         *shift(tube(0.036, 0.100, 20, caps=True), (-0.075, 0.0, 0.2755)),
         mat=mats["reservoir"])
    part("Crawler_Couplant_Cap",
         *shift(tube(0.015, 0.014, 14, caps=True), (-0.075, 0.0, 0.3325)),
         mat=mats["hub"])
    part("Crawler_Couplant_Strap",
         *shift(tube(0.040, 0.010, 20, caps=True), (-0.075, 0.0, 0.250)),
         mat=mats["frame"])
    part("Crawler_Couplant_Pump",
         *shift(tube(0.022, 0.048, 16, caps=True), (0.015, -0.052, 0.2495)),
         mat=mats["frame"])

    # Tank to pump, then forward and down the side of the probe. It ends at the
    # face, which is the correction: a line that stops beside the probe wets
    # the plate next to the measurement and not under it.
    # Tank, pump, then down the flank and round the outside of the deck
    # rather than through it, and back in underneath to the probe.
    part("Crawler_Couplant_Line",
         *sweep([(-0.075, -0.030, 0.230), (-0.040, -0.050, 0.240),
                 (0.000, -0.052, 0.248), (0.015, -0.052, 0.228),
                 (0.010, -0.110, 0.205), (-0.005, -0.165, 0.160),
                 (0.005, -0.205, 0.110), (0.035, -0.208, 0.070),
                 (0.065, -0.170, 0.058), (0.092, -0.110, 0.052),
                 (0.112, -0.055, 0.046), (0.127, -0.018, 0.034),
                 (0.130, 0.0, 0.018), (0.126, 0.0, 0.008)],
                0.0042, 8), mat=mats["couplant"])
    # The nozzle, turned in under the face.
    part("Crawler_Couplant_Nozzle",
         *sweep([(0.126, 0.0, 0.007), (0.118, 0.0, 0.0035),
                 (0.108, 0.0, 0.0028)], 0.0028, 6), mat=mats["hub"])

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

    rig.matrix_world = crawler_matrix(theta, z)
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

    # The crawler starts at the beginning of the run it has been sent to, and
    # the ripple detail covers exactly that run.
    run = next(s for s in seams if s["id"] == SURVEY_SEAM)
    window_lo, window_hi = survey_window()
    park_theta = window_lo
    park_z = run["z_m"]
    subject = next(d for d in defects if d.get("on_survey_run"))

    build_shell(mats, tank)
    build_fittings(mats, tank)
    detailed = build_seams(mats, tank, seams, rng, SURVEY_CENTRE_RAD, park_z,
                           defects)
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
        "survey_run": {"seam": SURVEY_SEAM, "z_m": run["z_m"],
                       "theta_from_rad": round(window_lo, 5),
                       "theta_to_rad": round(window_hi, 5),
                       "arc_m": SURVEY_ARC_M, "ripple_pitch_m": RIPPLE_PITCH,
                       "faults_on_it": [d["id"] for d in defects
                                        if d.get("on_survey_run")]},
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
        "survey_run_faults": [d["id"] for d in defects if d.get("on_survey_run")],
        "hidden_beside_the_run": [d["id"] for d in defects
                                  if d["kind"] == "thinning" and not d["visible"]
                                  and abs(d["z_m"] - park_z) < 1.2],
        "ground_truth": str(OUT),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
