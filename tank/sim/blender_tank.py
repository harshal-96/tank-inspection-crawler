"""Build the inside of a storage tank, its weld seams, its faults, and the crawler.

Run inside Blender:

    python scripts/blender_send.py tank/sim/blender_tank.py

Why a tank. A person enters a tank to look at its welds and its wall, and that
entry is what kills people: US confined spaces take about a hundred lives a
year, a quarter of them during inspection and cleaning, and most of the dead
are the second person who went in after the first. Emptying, cleaning and
gas-freeing the tank to allow that entry is what makes an internal inspection
take weeks rather than days. A crawler that goes in instead has to do three
things a camera alone cannot: find the seams, judge them, and decide where the
ultrasonic probe is worth putting, because a thirty metre wall cannot be probed
everywhere.

What this file makes, and why each piece is here.

  the shell         six courses, thick at the bottom and thin at the top, the
                    way a tank is actually built, so wall thickness is a real
                    function of height rather than one number.
  the seams         a circumferential seam between every pair of courses, a
                    vertical seam at every plate join, staggered course to
                    course as a real tank staggers them, and the chime weld
                    where the shell meets the floor, which is the one that
                    fails first.
  the faults        injected on purpose, each recorded before anything looks at
                    it, so every verdict can be marked against an answer
                    written in advance. Some are visible to a camera and some
                    are not: general wall loss shows nothing on the surface,
                    and that is the whole reason the probe has to be aimed
                    rather than swept.
  the crawler       a body on the wall with a camera, two lamps and an
                    ultrasonic probe under it. Inside a tank is pitch dark, so
                    the robot carries its own light and sees only what it
                    lights.

Ground truth goes to tank/data/tank_scene.json in metres, including the
thinning patches as parameters rather than a grid, so thickness can be asked
for at any point on the wall rather than at the points someone happened to
sample.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# --------------------------------------------------------------------- shape

RADIUS = 10.0            # metres, a 20 m diameter tank
COURSE_H = 2.0           # each ring of plates
COURSES = 6              # 12 m tall
PLATES_PER_COURSE = 8    # plate joins, so eight vertical seams a course
SHELL_SEGMENTS = 128     # facets around; a bead reads smooth at this count

# Plate thickness by course, bottom first. A tank carries its own head, so the
# bottom course is the thick one and the top course is the thin one.
COURSE_MM = (12.0, 11.0, 10.0, 9.0, 8.0, 6.0)
FLOOR_MM = 8.0

BEAD_R = 0.008           # weld bead stands 8 mm proud of the plate
SEED = 7

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "tank" / "data" / "tank_scene.json"

# ----------------------------------------------------------------- materials


def material(name: str, colour, metallic: float, roughness: float):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            node.inputs["Base Color"].default_value = (*colour, 1.0)
            node.inputs["Metallic"].default_value = metallic
            node.inputs["Roughness"].default_value = roughness
    return mat


def build_materials() -> dict:
    return {
        # Shot-blasted plate inside a tank is dark and dull, not showroom steel.
        "plate": material("tank_plate", (0.19, 0.20, 0.22), 0.85, 0.58),
        "bead": material("weld_bead", (0.42, 0.38, 0.30), 0.80, 0.42),
        "dark": material("defect_dark", (0.02, 0.02, 0.025), 0.10, 0.85),
        "rust": material("corrosion", (0.30, 0.13, 0.05), 0.15, 0.92),
        "robot": material("crawler_body", (0.75, 0.35, 0.05), 0.35, 0.45),
        "probe": material("ut_probe", (0.05, 0.06, 0.08), 0.60, 0.35),
    }


# --------------------------------------------------------------- mesh making
#
# Meshes are built from vertices rather than through bpy.ops, because ops need
# a live context and a selection state, and a build that half succeeds because
# the wrong object was selected is the worst kind of bug to find later.


def make(name: str, verts, faces, mat=None, smooth=True, collection=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if smooth and len(mesh.polygons):
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    obj = bpy.data.objects.new(name, mesh)
    if mat is not None:
        obj.data.materials.append(mat)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def tube(radius, height, segments=64, centre=(0.0, 0.0, 0.0), arc=(0.0, math.tau)):
    """An open cylinder about Z, for the shell and for vertical beads."""
    verts, faces = [], []
    closed = abs((arc[1] - arc[0]) - math.tau) < 1e-9
    rings = segments if closed else segments + 1
    for i in range(rings):
        t = arc[0] + (arc[1] - arc[0]) * (i / segments)
        x, y = radius * math.cos(t), radius * math.sin(t)
        verts.append((centre[0] + x, centre[1] + y, centre[2] - height / 2))
        verts.append((centre[0] + x, centre[1] + y, centre[2] + height / 2))
    for i in range(segments if closed else segments):
        a, b = 2 * i, 2 * i + 1
        c, d = (2 * ((i + 1) % rings)) + 1, 2 * ((i + 1) % rings)
        faces.append((a, b, c, d))
    return verts, faces


def ring(major_r, minor_r, centre=(0.0, 0.0, 0.0), major_seg=128, minor_seg=8,
         arc=(0.0, math.tau)):
    """A torus, used for every circumferential bead and for the manway rim."""
    verts, faces = [], []
    closed = abs((arc[1] - arc[0]) - math.tau) < 1e-9
    rings = major_seg if closed else major_seg + 1
    for i in range(rings):
        u = arc[0] + (arc[1] - arc[0]) * (i / major_seg)
        cu, su = math.cos(u), math.sin(u)
        for j in range(minor_seg):
            v = math.tau * j / minor_seg
            r = major_r + minor_r * math.cos(v)
            verts.append((centre[0] + r * cu, centre[1] + r * su,
                          centre[2] + minor_r * math.sin(v)))
    span = major_seg if closed else major_seg
    for i in range(span):
        n = (i + 1) % rings if closed else i + 1
        for j in range(minor_seg):
            k = (j + 1) % minor_seg
            faces.append((i * minor_seg + j, i * minor_seg + k,
                          n * minor_seg + k, n * minor_seg + j))
    return verts, faces


def box(size, centre=(0.0, 0.0, 0.0)):
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = centre
    verts = [(cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
             (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz),
             (cx - sx, cy - sy, cz + sz), (cx + sx, cy - sy, cz + sz),
             (cx + sx, cy + sy, cz + sz), (cx - sx, cy + sy, cz + sz)]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return verts, faces


def disc(radius, z=0.0, segments=128):
    verts = [(0.0, 0.0, z)]
    for i in range(segments):
        t = math.tau * i / segments
        verts.append((radius * math.cos(t), radius * math.sin(t), z))
    faces = [(0, i + 1, (i + 1) % segments + 1) for i in range(segments)]
    return verts, faces


def blob(radius, centre, segments=10):
    """A low sphere, for a pore or a corrosion pit."""
    verts, faces = [], []
    rows = segments // 2
    for i in range(rows + 1):
        phi = math.pi * i / rows
        for j in range(segments):
            th = math.tau * j / segments
            verts.append((centre[0] + radius * math.sin(phi) * math.cos(th),
                          centre[1] + radius * math.sin(phi) * math.sin(th),
                          centre[2] + radius * math.cos(phi)))
    for i in range(rows):
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((i * segments + j, i * segments + k,
                          (i + 1) * segments + k, (i + 1) * segments + j))
    return verts, faces


# --------------------------------------------------------------- the geometry


def look_at(eye, target, up_hint=(0.0, 0.0, 1.0)) -> Matrix:
    """A camera or spot matrix that points from eye to target.

    Blender cameras and spot lamps both look down their own -Z, so one helper
    aims both. The first build set these with euler angles on a rig that was
    itself rotated onto a curved wall, and every one of them ended up facing
    into the tank instead of at the plate.
    """
    eye, target = Vector(eye), Vector(target)
    forward = (target - eye).normalized()
    right = forward.cross(Vector(up_hint))
    if right.length < 1e-6:
        right = Vector((1.0, 0.0, 0.0))
    right.normalize()
    up = right.cross(forward)
    return Matrix((
        (right.x, up.x, -forward.x, eye.x),
        (right.y, up.y, -forward.y, eye.y),
        (right.z, up.z, -forward.z, eye.z),
        (0.0, 0.0, 0.0, 1.0),
    ))


def wall_point(theta: float, z: float, inset: float = 0.0):
    """A point on the inner face of the shell, inset metres in from the plate."""
    r = RADIUS - inset
    return (r * math.cos(theta), r * math.sin(theta), z)


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)


def collection(name: str):
    coll = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(coll)
    return coll


def build_shell(mats, coll):
    height = COURSE_H * COURSES
    verts, faces = tube(RADIUS, height, SHELL_SEGMENTS, centre=(0, 0, height / 2))
    make("Tank_Shell", verts, faces, mats["plate"], collection=coll)
    make("Tank_Floor", *disc(RADIUS, 0.0), mat=mats["plate"], smooth=False, collection=coll)
    # A cone roof, so an interior shot reads as a closed vessel rather than a pit.
    roof_v = [(0.0, 0.0, height + 1.2)]
    for i in range(SHELL_SEGMENTS):
        t = math.tau * i / SHELL_SEGMENTS
        roof_v.append((RADIUS * math.cos(t), RADIUS * math.sin(t), height))
    roof_f = [(0, i + 1, (i + 1) % SHELL_SEGMENTS + 1) for i in range(SHELL_SEGMENTS)]
    make("Tank_Roof", roof_v, roof_f, mats["plate"], collection=coll)


def build_seams(mats, coll) -> list[dict]:
    """Every weld the crawler could be sent to, with where it runs recorded."""
    seams: list[dict] = []

    # The chime: shell to floor. First to corrode, worst to lose.
    verts, faces = ring(RADIUS - BEAD_R, BEAD_R, centre=(0, 0, BEAD_R), major_seg=SHELL_SEGMENTS)
    make("Seam_Chime", verts, faces, mats["bead"], collection=coll)
    seams.append({"id": "CHIME", "kind": "chime", "z_m": round(BEAD_R, 4),
                  "length_m": round(math.tau * RADIUS, 3), "course": 0})

    # Circumferential seams between courses.
    for c in range(1, COURSES):
        z = c * COURSE_H
        verts, faces = ring(RADIUS - BEAD_R, BEAD_R, centre=(0, 0, z), major_seg=SHELL_SEGMENTS)
        make(f"Seam_C{c}", verts, faces, mats["bead"], collection=coll)
        seams.append({"id": f"C{c}", "kind": "circumferential", "z_m": z,
                      "length_m": round(math.tau * RADIUS, 3), "course": c})

    # Vertical plate joins, staggered half a plate each course so no two
    # verticals line up across a horizontal weld, exactly as a tank is built.
    step = math.tau / PLATES_PER_COURSE
    for c in range(COURSES):
        z0 = c * COURSE_H
        for p in range(PLATES_PER_COURSE):
            theta = p * step + (step / 2 if c % 2 else 0.0)
            verts, faces = tube(BEAD_R, COURSE_H, 12, centre=(0, 0, 0))
            obj = make(f"Seam_V{c}_{p}", verts, faces, mats["bead"], collection=coll)
            obj.location = wall_point(theta, z0 + COURSE_H / 2, inset=BEAD_R)
            seams.append({"id": f"V{c}_{p}", "kind": "vertical", "course": c,
                          "theta_rad": round(theta, 5),
                          "z_from_m": z0, "z_to_m": z0 + COURSE_H,
                          "length_m": COURSE_H})
    return seams


def build_manway(mats, coll):
    """Where the crawler goes in, and the only daylight in the tank."""
    theta, z = 0.0, 1.1
    verts, faces = ring(0.35, 0.05, centre=(0, 0, 0), major_seg=48, minor_seg=8)
    rim = make("Manway_Rim", verts, faces, mats["bead"], collection=coll)
    rim.location = wall_point(theta, z, inset=0.04)
    rim.rotation_euler = (math.pi / 2, 0.0, theta)

    light = bpy.data.lights.new("Manway_Daylight", type="AREA")
    light.energy = 220.0
    light.size = 0.7
    obj = bpy.data.objects.new("Manway_Daylight", light)
    obj.location = wall_point(theta, z, inset=0.6)
    obj.rotation_euler = (math.pi / 2, 0.0, theta + math.pi / 2)
    coll.objects.link(obj)


# ----------------------------------------------------------------- the faults


def inject_defects(mats, coll, seams, rng) -> list[dict]:
    """Faults put in on purpose, each written down before anything looks at it.

    Visible faults sit on or beside a weld and a camera can find them. Thinning
    does not: the wall is simply less metal than it was, with nothing on the
    surface to see. Half the thinning here is given away by pitting on top of
    it and half is not, which is what stops a probe planner from getting a good
    score by following the camera alone.
    """
    defects: list[dict] = []
    circ = [s for s in seams if s["kind"] in ("circumferential", "chime")]
    vert = [s for s in seams if s["kind"] == "vertical"]

    def record(kind, seam_id, theta, z, **extra):
        item = {"id": f"D{len(defects) + 1:03d}", "kind": kind, "seam": seam_id,
                "theta_rad": round(theta, 5), "z_m": round(z, 4),
                "centre_m": [round(v, 4) for v in wall_point(theta, z, BEAD_R)]}
        item.update(extra)
        defects.append(item)
        return item

    # Undercut: a groove melted into the plate along the toe of the weld, so it
    # runs beside the bead, not through it.
    for _ in range(4):
        seam = rng.choice(circ)
        theta = rng.uniform(0, math.tau)
        length = rng.uniform(0.25, 0.9)
        depth_mm = rng.uniform(0.8, 2.2)
        span = length / RADIUS
        verts, faces = ring(RADIUS - 0.004, 0.0035, centre=(0, 0, seam["z_m"] + BEAD_R * 1.7),
                            major_seg=max(8, int(span / math.tau * SHELL_SEGMENTS)),
                            minor_seg=6, arc=(theta, theta + span))
        make(f"Defect_undercut_{seam['id']}_{len(defects)}", verts, faces,
             mats["dark"], collection=coll)
        record("undercut", seam["id"], theta + span / 2, seam["z_m"],
               length_mm=round(length * 1000, 1), depth_mm=round(depth_mm, 2), visible=True)

    # Porosity: pores in the bead face, several small ones together.
    for _ in range(4):
        seam = rng.choice(circ + vert)
        if seam["kind"] == "vertical":
            theta = seam["theta_rad"]
            z0 = rng.uniform(seam["z_from_m"] + 0.2, seam["z_to_m"] - 0.2)
        else:
            theta, z0 = rng.uniform(0, math.tau), seam["z_m"]
        count = rng.randint(4, 9)
        for k in range(count):
            r_mm = rng.uniform(1.2, 3.0)
            if seam["kind"] == "vertical":
                pos = wall_point(theta, z0 + rng.uniform(-0.12, 0.12), BEAD_R * 0.7)
            else:
                pos = wall_point(theta + rng.uniform(-0.012, 0.012), z0, BEAD_R * 0.7)
            make(f"Defect_pore_{len(defects)}_{k}", *blob(r_mm / 1000, pos),
                 mat=mats["dark"], collection=coll)
        record("porosity", seam["id"], theta, z0, count=count, visible=True)

    # A crack across the bead. The one that matters and the one that is thin.
    for _ in range(3):
        seam = rng.choice(circ)
        theta = rng.uniform(0, math.tau)
        length_mm = rng.uniform(25, 90)
        verts, faces = box((0.0015, 0.0015, length_mm / 1000))
        obj = make(f"Defect_crack_{len(defects)}", verts, faces, mats["dark"],
                   smooth=False, collection=coll)
        obj.location = wall_point(theta, seam["z_m"], BEAD_R * 0.6)
        record("crack", seam["id"], theta, seam["z_m"],
               length_mm=round(length_mm, 1), visible=True)

    # Lack of fusion: the bead stops and starts, leaving bare joint.
    for _ in range(3):
        seam = rng.choice(vert)
        theta = seam["theta_rad"]
        z0 = rng.uniform(seam["z_from_m"] + 0.3, seam["z_to_m"] - 0.3)
        gap = rng.uniform(0.06, 0.16)
        verts, faces = box((0.004, 0.004, gap))
        obj = make(f"Defect_lof_{len(defects)}", verts, faces, mats["dark"],
                   smooth=False, collection=coll)
        obj.location = wall_point(theta, z0, BEAD_R * 0.4)
        record("lack_of_fusion", seam["id"], theta, z0,
               length_mm=round(gap * 1000, 1), visible=True)

    # Wall loss. Three patches wear pitting a camera can see, three do not.
    for index in range(6):
        visible = index < 3
        theta = rng.uniform(0, math.tau)
        course = rng.randint(0, COURSES - 1)
        z = rng.uniform(course * COURSE_H + 0.25, (course + 1) * COURSE_H - 0.25)
        patch_r = rng.uniform(0.25, 0.7)
        loss_mm = rng.uniform(1.5, 4.5)
        nominal = COURSE_MM[course]
        if visible:
            for k in range(rng.randint(14, 28)):
                a = rng.uniform(0, math.tau)
                d = rng.uniform(0, patch_r) ** 0.5
                pos = wall_point(theta + math.cos(a) * d / RADIUS,
                                 z + math.sin(a) * d, 0.004)
                make(f"Defect_pit_{index}_{k}", *blob(rng.uniform(0.004, 0.013), pos),
                     mat=mats["rust"], collection=coll)
        record("thinning", None, theta, z, patch_radius_m=round(patch_r, 3),
               loss_mm=round(loss_mm, 2), nominal_mm=nominal,
               remaining_mm=round(nominal - loss_mm, 2),
               falloff="cosine", visible=visible)
    return defects


# ---------------------------------------------------------------- the crawler


def build_crawler(mats, coll, theta: float, z: float):
    """A body on the wall, a camera, two lamps and a probe under it.

    The robot clings to the inside of the shell, so the direction it calls down
    points outward through the plate, and its camera looks the same way. Parts
    are built in the robot's own frame and the whole assembly is placed with
    one matrix, so moving it along a seam later is one line rather than twenty.
    """
    rig = bpy.data.objects.new("Crawler", None)
    rig.empty_display_size = 0.15
    coll.objects.link(rig)

    def part(name, verts, faces, mat, smooth=True):
        obj = make(name, verts, faces, mat, smooth=smooth, collection=coll)
        obj.parent = rig
        return obj

    # Body sits 60 mm off the plate on its wheels; +Z is away from the wall.
    part("Crawler_Body", *box((0.42, 0.30, 0.13), (0.0, 0.0, 0.115)),
         mat=mats["robot"], smooth=False)
    for sx in (-0.16, 0.16):
        for sy in (-0.15, 0.15):
            verts, faces = tube(0.055, 0.045, 20, centre=(0, 0, 0))
            wheel = part(f"Crawler_Wheel_{sx:+.2f}_{sy:+.2f}", verts, faces, mats["dark"])
            wheel.location = (sx, sy, 0.055)
            wheel.rotation_euler = (math.pi / 2, 0.0, 0.0)

    # The ultrasonic probe rides under the body, on the plate.
    verts, faces = tube(0.022, 0.05, 16, centre=(0, 0, 0))
    probe = part("Crawler_UT_Probe", verts, faces, mats["probe"])
    probe.location = (-0.12, 0.0, 0.03)

    # The camera rides at the front looking forward and down onto the plate, so
    # the seam the robot is straddling runs away from it up the frame. In the
    # robot's own frame the plate is Z = 0 and travel is +X, so "look at a spot
    # on the plate a little way ahead" is literally that.
    cam_data = bpy.data.cameras.new("Crawler_Cam")
    # 25 mm at 220 mm off the plate covers about 300 mm of seam, so an 8 mm
    # bead is eighty pixels across and a 2 mm pore is still ten. A wider lens
    # looking along the wall put the bead on the horizon, where no defect in it
    # has a shape.
    cam_data.lens = 25.0
    cam = bpy.data.objects.new("Crawler_Cam", cam_data)
    cam.parent = rig
    coll.objects.link(cam)
    # Straight down at the plate, on a short boom ahead of the body so the body
    # is not in its own shot. Aimed forward along the seam instead, the bead
    # ran to a vanishing point and no fault in it had a shape worth judging.
    cam.matrix_basis = look_at((0.46, 0.0, 0.24), (0.46, 0.0, 0.0),
                               up_hint=(0.0, 1.0, 0.0))

    # Two lamps either side of the camera, throwing light forward along the
    # seam. A bead is read by the shadow it casts, so the light comes from
    # behind the camera and across, never straight down the lens.
    # Watts, at a quarter of a metre. The first build used 260 and returned a
    # white frame: a lamp that close needs single figures, and the light has to
    # come from the side so a bead casts a shadow instead of glaring back.
    for sy in (-0.20, 0.20):
        lamp = bpy.data.lights.new(f"Crawler_Lamp_{sy:+.2f}", type="SPOT")
        lamp.energy = 6.0
        lamp.spot_size = math.radians(120)
        lamp.spot_blend = 0.6
        lamp.shadow_soft_size = 0.015
        obj = bpy.data.objects.new(lamp.name, lamp)
        obj.parent = rig
        coll.objects.link(obj)
        # Either side of the camera and low, so the light crosses the bead and
        # a pore or an undercut throws a shadow. Lit down the lens they vanish.
        obj.matrix_basis = look_at((0.46, sy, 0.14), (0.46, 0.0, 0.0),
                                   up_hint=(1.0, 0.0, 0.0))

    # Place the whole rig on the wall: +X along the seam, +Z off the plate.
    normal = Vector((-math.cos(theta), -math.sin(theta), 0.0))   # inward
    along = Vector((-math.sin(theta), math.cos(theta), 0.0))     # tangential
    across = normal.cross(along)
    p = Vector(wall_point(theta, z))
    rig.matrix_world = Matrix((
        (along.x, across.x, normal.x, p.x),
        (along.y, across.y, normal.y, p.y),
        (along.z, across.z, normal.z, p.z),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return rig, cam


def build_overview_camera(coll, rig):
    """Over the robot's shoulder, looking down the seam it is following.

    Built in the robot's own frame and then carried onto the wall by the rig's
    matrix, so it stays behind the robot wherever the robot is parked.
    """
    cam_data = bpy.data.cameras.new("Overview_Cam")
    cam_data.lens = 30.0
    cam = bpy.data.objects.new("Overview_Cam", cam_data)
    coll.objects.link(cam)
    local = look_at((-1.15, 0.85, 0.95), (1.30, 0.0, 0.0))
    cam.matrix_world = rig.matrix_world @ local
    return cam


def set_render(scene, camera):
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.camera = camera
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
            # Not quite black: a sealed tank has some bounce, and pure black
            # renders hide whether the geometry is right.
            node.inputs[0].default_value = (0.020, 0.022, 0.027, 1.0)
            node.inputs[1].default_value = 1.0


# -------------------------------------------------------------------- the run


def main() -> dict:
    rng = random.Random(SEED)
    clear_scene()
    mats = build_materials()
    shell = collection("Tank")
    faults = collection("Defects")
    robot = collection("Crawler")

    build_shell(mats, shell)
    seams = build_seams(mats, shell)
    build_manway(mats, shell)
    defects = inject_defects(mats, faults, seams, rng)

    # Parked on a fault rather than at a tidy round number, so what the camera
    # sees is something the ground truth already has an answer for.
    subject = next((d for d in defects
                    if d["kind"] in ("porosity", "undercut", "crack")
                    and d["seam"] and d["seam"].startswith("C")), defects[0])
    park_theta = subject["theta_rad"] - 0.46 / RADIUS
    park_z = subject["z_m"]
    rig, crawler_cam = build_crawler(mats, robot, park_theta, park_z)
    overview = build_overview_camera(robot, rig)
    set_render(bpy.context.scene, overview)

    ground_truth = {
        "units": "metres unless a key says mm",
        "seed": SEED,
        "tank": {"radius_m": RADIUS, "height_m": COURSE_H * COURSES,
                 "courses": COURSES, "course_height_m": COURSE_H,
                 "plates_per_course": PLATES_PER_COURSE,
                 "course_thickness_mm": list(COURSE_MM), "floor_thickness_mm": FLOOR_MM},
        "seams": seams,
        "defects": defects,
        "crawler": {"parked_at": {"theta_rad": round(park_theta, 5), "z_m": park_z},
                    "looking_at": subject["id"],
                    "camera": crawler_cam.name, "probe": "Crawler_UT_Probe"},
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
        "seams": {"total": len(seams),
                  "circumferential": sum(1 for s in seams if s["kind"] == "circumferential"),
                  "vertical": sum(1 for s in seams if s["kind"] == "vertical"),
                  "chime": sum(1 for s in seams if s["kind"] == "chime")},
        "seam_metres": round(sum(s["length_m"] for s in seams), 1),
        "defects": kinds,
        "hidden_thinning": sum(1 for d in defects
                               if d["kind"] == "thinning" and not d["visible"]),
        "parked_on": {"defect": subject["id"], "kind": subject["kind"],
                      "seam": subject["seam"]},
        "ground_truth": str(OUT),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
