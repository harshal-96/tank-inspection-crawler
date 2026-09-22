"""Shapes, surfaces and framing shared by every scene in this project.

Two robots are modelled here: the crawler that goes into a tank, and the bench
rig on the parts list that runs the same software on a plate of real steel. One
is what the work scales to and the other is what can be built this month, and
they have to be built from the same primitives or the comparison between them
is a comparison of two modelling styles.

Meshes are built from vertices rather than through bpy.ops, because ops need a
live context and a selection state, and a build that half succeeds because the
wrong object was selected is the worst kind of bug to find later.
"""

from __future__ import annotations

import math

import bpy
from mathutils import Matrix, Vector


def _principled(mat):
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def _coords(mat):
    """Object-space coordinates, so a noise scale means something in metres."""
    node = mat.node_tree.nodes.new("ShaderNodeTexCoord")
    node.location = (-1100, 0)
    return node.outputs["Object"]


def steel_material(name, colour, roughness, metallic, scale, bump, fine=None):
    """Rolled plate: mottled, faintly dimpled, never a flat colour.

    Two scales of relief, because one is always wrong somewhere. The coarse one
    carries the rolling and the pitting a metre away; the fine one is the shot
    blast, and without it a close-up of a weld is moulded plastic.
    """
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = _principled(mat)
    coords = _coords(mat)

    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.location = (-850, -250)
    grain.inputs["Scale"].default_value = scale
    grain.inputs["Detail"].default_value = 9.0
    grain.inputs["Roughness"].default_value = 0.65
    nt.links.new(coords, grain.inputs["Vector"])

    relief = nt.nodes.new("ShaderNodeBump")
    relief.location = (-450, -350)
    relief.inputs["Strength"].default_value = bump
    nt.links.new(grain.outputs["Fac"], relief.inputs["Height"])

    blast = nt.nodes.new("ShaderNodeTexNoise")
    blast.location = (-650, -560)
    blast.inputs["Scale"].default_value = (fine or scale * 14.0)
    blast.inputs["Detail"].default_value = 6.0
    nt.links.new(coords, blast.inputs["Vector"])

    fine_relief = nt.nodes.new("ShaderNodeBump")
    fine_relief.location = (-260, -450)
    fine_relief.inputs["Strength"].default_value = min(0.5, bump * 1.4)
    nt.links.new(blast.outputs["Fac"], fine_relief.inputs["Height"])
    nt.links.new(relief.outputs["Normal"], fine_relief.inputs["Normal"])
    nt.links.new(fine_relief.outputs["Normal"], bsdf.inputs["Normal"])

    blotch = nt.nodes.new("ShaderNodeTexNoise")
    blotch.location = (-850, 200)
    blotch.inputs["Scale"].default_value = scale * 0.06
    blotch.inputs["Detail"].default_value = 4.0
    nt.links.new(coords, blotch.inputs["Vector"])

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.location = (-600, 200)
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[0].color = (*[c * 0.80 for c in colour], 1.0)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (*[min(1.0, c * 1.14) for c in colour], 1.0)
    nt.links.new(blotch.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    rough = nt.nodes.new("ShaderNodeMapRange")
    rough.location = (-450, -80)
    rough.inputs["To Min"].default_value = max(0.0, roughness - 0.14)
    rough.inputs["To Max"].default_value = min(1.0, roughness + 0.14)
    nt.links.new(grain.outputs["Fac"], rough.inputs["Value"])
    nt.links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])

    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def emissive(name, colour, strength):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _principled(mat)
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    for key in ("Emission Color", "Emission"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = (*colour, 1.0)
            break
    if "Emission Strength" in bsdf.inputs:
        bsdf.inputs["Emission Strength"].default_value = strength
    return mat


def plain(name, colour, metallic, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = _principled(mat)
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make(name, verts, faces, mat=None, smooth=True, collection=None):
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


def merge(parts):
    """Stitch many vertex and face pairs into one, so a bead is one object."""
    verts, faces = [], []
    for v, f in parts:
        base = len(verts)
        verts.extend(v)
        faces.extend(tuple(i + base for i in face) for face in f)
    return verts, faces


def shift(part, offset):
    verts, faces = part
    dx, dy, dz = offset
    return [(x + dx, y + dy, z + dz) for x, y, z in verts], faces


def tube(radius, height, segments=64, centre=(0.0, 0.0, 0.0), arc=(0.0, math.tau),
         caps=False):
    verts, faces = [], []
    closed = abs((arc[1] - arc[0]) - math.tau) < 1e-9
    rings = segments if closed else segments + 1
    for i in range(rings):
        t = arc[0] + (arc[1] - arc[0]) * (i / segments)
        x, y = radius * math.cos(t), radius * math.sin(t)
        verts.append((centre[0] + x, centre[1] + y, centre[2] - height / 2))
        verts.append((centre[0] + x, centre[1] + y, centre[2] + height / 2))
    for i in range(segments):
        n = (i + 1) % rings
        faces.append((2 * i, 2 * i + 1, 2 * n + 1, 2 * n))
    if caps and closed:
        lo = len(verts)
        verts.append((centre[0], centre[1], centre[2] - height / 2))
        hi = len(verts)
        verts.append((centre[0], centre[1], centre[2] + height / 2))
        for i in range(segments):
            n = (i + 1) % segments
            faces.append((lo, 2 * n, 2 * i))
            faces.append((hi, 2 * i + 1, 2 * n + 1))
    return verts, faces


def ring(major_r, minor_r, centre=(0.0, 0.0, 0.0), major_seg=128, minor_seg=8,
         arc=(0.0, math.tau)):
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
    for i in range(major_seg):
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


def chamfered(size, centre=(0.0, 0.0, 0.0), bevel=0.02):
    """A box with its top edges taken off, which is what built kit looks like."""
    sx, sy, sz = (s / 2 for s in size)
    cx, cy, cz = centre
    b = bevel
    lo = [(cx - sx, cy - sy, cz - sz), (cx + sx, cy - sy, cz - sz),
          (cx + sx, cy + sy, cz - sz), (cx - sx, cy + sy, cz - sz)]
    mid = [(cx - sx, cy - sy, cz + sz - b), (cx + sx, cy - sy, cz + sz - b),
           (cx + sx, cy + sy, cz + sz - b), (cx - sx, cy + sy, cz + sz - b)]
    top = [(cx - sx + b, cy - sy + b, cz + sz), (cx + sx - b, cy - sy + b, cz + sz),
           (cx + sx - b, cy + sy - b, cz + sz), (cx - sx + b, cy + sy - b, cz + sz)]
    verts = lo + mid + top
    faces = [(0, 1, 2, 3), (8, 11, 10, 9)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append((i, 4 + i, 4 + j, j))
        faces.append((4 + i, 8 + i, 8 + j, 4 + j))
    return verts, faces


def disc(radius, z=0.0, segments=128):
    verts = [(0.0, 0.0, z)]
    for i in range(segments):
        t = math.tau * i / segments
        verts.append((radius * math.cos(t), radius * math.sin(t), z))
    faces = [(0, i + 1, (i + 1) % segments + 1) for i in range(segments)]
    return verts, faces


def ellipsoid(centre, axes, frame=None, segments=10):
    """A squashed sphere in any frame: one weld pool, one pore, one pit."""
    ex, ey, ez = frame or (Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)))
    c = Vector(centre)
    verts, faces = [], []
    rows = max(3, segments // 2)
    for i in range(rows + 1):
        phi = math.pi * i / rows
        for j in range(segments):
            th = math.tau * j / segments
            p = (c + ex * (axes[0] * math.sin(phi) * math.cos(th))
                 + ey * (axes[1] * math.sin(phi) * math.sin(th))
                 + ez * (axes[2] * math.cos(phi)))
            verts.append((p.x, p.y, p.z))
    for i in range(rows):
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((i * segments + j, i * segments + k,
                          (i + 1) * segments + k, (i + 1) * segments + j))
    return verts, faces


def sweep(points, radius, segments=8):
    """A tube swept along a polyline, for the tether and the couplant hose."""
    pts = [Vector(p) for p in points]
    verts, faces = [], []
    for i, p in enumerate(pts):
        ahead = pts[min(i + 1, len(pts) - 1)]
        back = pts[max(i - 1, 0)]
        tangent = ahead - back
        tangent = tangent.normalized() if tangent.length > 1e-9 else Vector((1, 0, 0))
        helper = Vector((0, 0, 1)) if abs(tangent.z) < 0.9 else Vector((1, 0, 0))
        u = tangent.cross(helper).normalized()
        v = tangent.cross(u)
        for j in range(segments):
            a = math.tau * j / segments
            q = p + u * (radius * math.cos(a)) + v * (radius * math.sin(a))
            verts.append((q.x, q.y, q.z))
    for i in range(len(pts) - 1):
        for j in range(segments):
            k = (j + 1) % segments
            faces.append((i * segments + j, i * segments + k,
                          (i + 1) * segments + k, (i + 1) * segments + j))
    return verts, faces


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


def rounded_plate(size, centre=(0.0, 0.0, 0.0), radius=0.006, corner_steps=5):
    """A flat plate with rounded corners, which is what a cut chassis looks like."""
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = centre
    outline = []
    for ox, oy, start in ((sx - radius, sy - radius, 0.0),
                          (-(sx - radius), sy - radius, math.pi / 2),
                          (-(sx - radius), -(sy - radius), math.pi),
                          (sx - radius, -(sy - radius), 3 * math.pi / 2)):
        for i in range(corner_steps + 1):
            a = start + (math.pi / 2) * i / corner_steps
            outline.append((cx + ox + radius * math.cos(a),
                            cy + oy + radius * math.sin(a)))
    verts, faces = [], []
    n = len(outline)
    for x, y in outline:
        verts.append((x, y, cz - sz))
    for x, y in outline:
        verts.append((x, y, cz + sz))
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    faces.append(tuple(range(n - 1, -1, -1)))
    faces.append(tuple(range(n, 2 * n)))
    return verts, faces


def bead_profile(rng, arc_len, pitch):
    """One pool's worth of jitter for every pool along a run of weld."""
    return [rng.uniform(0.80, 1.20) for _ in range(int(arc_len / pitch) + 3)]


def bead_surface(place, arc_len, jitter, minor_r, pitch, around=12, per_ripple=5,
                 amp=0.17):
    """Sweep a half-tube along a weld, with the crest rippled into crescents.

    A bead is one continuous run of metal, not a row of beads on a necklace,
    which is what a chain of separate ellipsoids gave: every narrow pool left a
    notch a camera read as a break. So the surface is swept once and its radius
    modulated, and the phase of that modulation is shifted across the width so
    each crest curves back the way a real pool freezes behind the arc.

    place(fraction, across_m, out_m) puts a point on the wall, so the same
    sweep serves a circumferential seam and a vertical one.
    """
    steps = max(8, int(arc_len / pitch * per_ripple))
    width, crown = minor_r * 1.35, minor_r * 1.05
    verts, faces = [], []
    for i in range(steps + 1):
        f = i / steps
        s_m = f * arc_len
        k = s_m / pitch
        crest = amp * jitter[min(int(k), len(jitter) - 1)]
        wander = 0.0011 * math.sin(s_m * 2.1) + 0.0005 * math.sin(s_m * 7.3)
        for j in range(around + 1):
            v = math.pi * j / around
            # The crest lags at the toes and leads at the crown, which is what
            # makes a stacked-dime weld look like crescents rather than rings.
            ripple = 1.0 + crest * math.sin(math.tau * (k + 0.34 * math.cos(v)))
            verts.append(place(f, math.cos(v) * width * ripple + wander,
                               math.sin(v) * crown * ripple))
    for i in range(steps):
        for j in range(around):
            base = i * (around + 1) + j
            faces.append((base, base + 1, base + around + 2, base + around + 1))
    return verts, faces


# A camera on a robot looks at things a hand's width away, and Blender's
# default near clip is 100 mm. The first bench render was a brown frame with
# nothing in it: the plate, the weld and the bench were all inside the clip
# plane and none of them was drawn.


def camera(name, lens_mm, collection=None, clip_start=0.004, clip_end=300.0):
    """A camera that can see something held close to it."""
    data = bpy.data.cameras.new(name)
    data.lens = lens_mm
    data.clip_start = clip_start
    data.clip_end = clip_end
    obj = bpy.data.objects.new(name, data)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj
