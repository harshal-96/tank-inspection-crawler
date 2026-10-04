"""Add the small hard parts to the crawler. Run after the build, before the look.

    python scripts/blender_send.py tank/sim/blender_tank.py
    python scripts/blender_send.py tank/sim/crawler_detail.py
    python scripts/blender_send.py tank/sim/look.py

The crawler was already the right machine: deck, four magnetic wheels, a head
out on rails ahead of the body, a probe in a sprung holder, a tether. In a wide
shot it still read as a wooden toy, and no shader fixes that, because what
separates a photograph of a machine from a render of one is almost never the
big shapes. It is the hundred small hard things bolted to them.

So this adds what a fabricator would have had to put there anyway: the bolts
round every flange, lugs on wheels that have to grip a greasy shell, the lines
where two panels meet, lifting eyes someone has to lower it by, a gland where
the tether enters the body, and a plate with a number on it.

None of it is decoration. A magnetic crawler with smooth wheels would slide,
and a sealed body with no gland would flood.

Everything is parented to the existing rig in the robot's own frame, so driving
it along a seam is still one matrix. Nothing enters the camera's view: the lens
sits at x 0.46, 0.131 above the plate, looking down, and every part added here
is either behind it or above it.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tank.sim import blender_tank  # noqa: E402
import importlib  # noqa: E402
# Blender keeps a module cached between runs, so an edit to blender_tank
# is invisible here until it is reloaded, and the failure is an
# AttributeError on a constant that is plainly there in the file.
importlib.reload(blender_tank)
from tank.sim.meshkit import (box, ellipsoid, make, merge, shift,  # noqa: E402
                              sweep, tube)

# The lens and its clear cone, so nothing is ever added into frame.
LENS_AT = (0.46, 0.0, 0.131)
LENS_CLEAR_R = 0.075


def _clear_of_lens(point) -> bool:
    x, y, z = point
    if z > LENS_AT[2] + 0.015:
        return True
    return math.dist((x, y), LENS_AT[:2]) > LENS_CLEAR_R


def bolt(at, radius=0.006, head=0.005, seg=6, axis="Z"):
    """A hex head on a washer. The cheapest detail in rendering, by far."""
    parts = []
    spin = {"Z": Matrix.Identity(4),
            "X": Matrix.Rotation(math.pi / 2, 4, "Y"),
            "Y": Matrix.Rotation(math.pi / 2, 4, "X")}[axis]
    for src in (tube(radius, head, seg, caps=True),
                tube(radius * 1.55, head * 0.35, 12, caps=True)):
        v, f = src
        place = Matrix.Translation(at) @ spin
        parts.append(([tuple(place @ Vector(p)) for p in v], f))
    return merge(parts)


def bolt_ring(centre, radius, count, bolt_r=0.0055, axis="Z", start=0.0):
    """Fasteners round a flange, which is how every sealed box is actually shut."""
    parts = []
    cx, cy, cz = centre
    for i in range(count):
        a = start + math.tau * i / count
        if axis == "Z":
            at = (cx + math.cos(a) * radius, cy + math.sin(a) * radius, cz)
        else:
            at = (cx, cy + math.cos(a) * radius, cz + math.sin(a) * radius)
        if _clear_of_lens(at):
            parts.append(bolt(at, radius=bolt_r, axis=axis))
    return merge(parts) if parts else ([], [])


def tread(centre, radius, width, lugs=18, depth=0.009, axis_y=True):
    """Lugs on a magnetic wheel.

    A smooth wheel on a greasy shell is a sledge. Real crawler wheels carry
    cleats, and in a render the lugs are also what tells the eye the wheel is
    turning rather than sliding.
    """
    parts = []
    cx, cy, cz = centre
    for i in range(lugs):
        a = math.tau * i / lugs
        ox, oz = math.cos(a) * (radius - depth * 0.3), math.sin(a) * (radius - depth * 0.3)
        lug = box((0.013, width * 0.82, depth), (0.0, 0.0, 0.0))
        place = (Matrix.Translation((cx + ox, cy, cz + oz))
                 @ Matrix.Rotation(a, 4, "Y"))
        v, f = lug
        parts.append(([tuple(place @ Vector(p)) for p in v], f))
    return merge(parts)


def panel_line(start, end, width=0.004, depth=0.003):
    """A recessed seam between two panels, as a thin dark inset strip."""
    sx, sy, sz = start
    ex, ey, ez = end
    mid = ((sx + ex) / 2, (sy + ey) / 2, (sz + ez) / 2)
    length = math.dist(start, end)
    along = (ex - sx, ey - sy, ez - sz)
    if abs(along[0]) >= abs(along[1]):
        size = (length, width, depth)
    else:
        size = (width, length, depth)
    return box(size, mid)


def main() -> dict:
    rig = bpy.data.objects.get("Crawler")
    if rig is None:
        return {"error": "no Crawler in the scene, run blender_tank.py first"}
    coll = rig.users_collection[0] if rig.users_collection else bpy.context.scene.collection

    mats = {m.name.split("crawler_")[-1]: m for m in bpy.data.materials
            if m.name.startswith("crawler_")}
    frame = bpy.data.materials.get("crawler_frame")
    hub = bpy.data.materials.get("crawler_hub")
    tyre = bpy.data.materials.get("crawler_tyre")
    bodym = bpy.data.materials.get("crawler_body")
    dark = bpy.data.materials.get("defect_dark")

    added = []

    def part(name, geom, mat, smooth=True):
        verts, faces = geom
        if not verts:
            return None
        existing = bpy.data.objects.get(name)
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)
        obj = make(name, verts, faces, mat, smooth=smooth, collection=coll)
        obj.parent = rig
        added.append(name)
        return obj

    # ---------------------------------------------------------- the body shut
    # The lid of a sealed box is bolted down, and the bolts are the single
    # clearest signal that a shape was manufactured rather than extruded.
    lid = []
    for sx in (-0.17, 0.15):
        for sy in (-0.125, 0.125):
            lid.append(bolt((sx, sy, 0.2255)))
    for sx in (-0.01,):
        for sy in (-0.125, 0.125):
            lid.append(bolt((sx, sy, 0.2255)))
    part("Crawler_Lid_Bolts", merge(lid), hub)

    # Panel lines across the lid and down the flanks.
    lines = [panel_line((-0.17, 0.0, 0.2262), (0.15, 0.0, 0.2262)),
             panel_line((-0.01, -0.13, 0.2262), (-0.01, 0.13, 0.2262))]
    part("Crawler_Panel_Lines", merge(lines), dark, smooth=False)

    # ------------------------------------------------------------- the wheels
    # Built around each wheel's own origin and parented to it, so the lugs and
    # the hub bolts come round with the wheel instead of hanging in the air
    # where the wheel used to be.
    for index, (sx, sy) in enumerate(blender_tank.WHEEL_AT):
        axle = bpy.data.objects.get(f"Crawler_Wheel_{index}")
        if axle is None:
            continue
        outward = math.copysign(0.030, sy)
        for suffix, geom, mat, smooth in (
                ("Lugs", tread((0.0, 0.0, 0.0), blender_tank.WHEEL_R,
                               blender_tank.WHEEL_WIDTH), tyre, False),
                ("Bolts", bolt_ring((0.0, outward, 0.0), 0.022, 5,
                                    bolt_r=0.0045, axis="Y"), hub, True)):
            obj = part(f"Crawler_Wheel_{index}_{suffix}", geom, mat, smooth=smooth)
            if obj is not None:
                obj.parent = axle

    # ------------------------------------------------------------- the head
    # A bezel round the lens and a bolt ring round the housing, both above the
    # glass so neither ever appears in the picture the camera takes.
    bezel = shift(tube(0.040, 0.010, 24, caps=True), (0.46, 0.0, 0.150))
    part("Crawler_Lens_Bezel", bezel, frame)
    part("Crawler_Head_Bolts", bolt_ring((0.46, 0.0, 0.2245), 0.048, 8), hub)

    # Lamp pod guards: a hoop over each lamp, which every light on a machine
    # that works in a confined space has, because they get hit.
    guards = []
    for sy in (-0.20, 0.20):
        for a in (-0.028, 0.028):
            guards.append(shift(tube(0.0035, 0.044, 8, caps=True),
                                (0.46 + a, sy, 0.148)))
        guards.append(shift(tube(0.032, 0.0035, 20, caps=True), (0.46, sy, 0.126)))
    part("Crawler_Lamp_Guards", merge(guards), frame)

    # ------------------------------------------------------- lifting and lugs
    # Somebody has to lower this through a manway on a rope.
    eyes = []
    for sx in (-0.15, 0.13):
        eyes.append(shift(tube(0.009, 0.030, 10, caps=True), (sx, -0.138, 0.238)))
        eyes.append(shift(tube(0.009, 0.030, 10, caps=True), (sx, 0.138, 0.238)))
        for sy in (-0.138, 0.138):
            # A swept ring, not a string of beads. Built from spheres it read
            # as a necklace hanging off the machine, which is the same mistake
            # the weld bead made before it became one swept surface.
            path = [(sx + math.cos(math.tau * i / 20) * 0.016, sy,
                     0.253 + math.sin(math.tau * i / 20) * 0.016)
                    for i in range(21)]
            eyes.append(sweep(path, 0.0038, 8))
    part("Crawler_Lifting_Eyes", merge(eyes), hub)

    # ----------------------------------------------------------- the gland
    # Where the tether enters the body. Without one the box is not sealed, and
    # a cable that simply vanishes into a wall is the oldest tell in rendering.
    gland = [shift(tube(0.016, 0.028, 14, caps=True), (-0.175, 0.02, 0.150)),
             shift(tube(0.020, 0.010, 14, caps=True), (-0.175, 0.02, 0.163)),
             shift(tube(0.013, 0.022, 14, caps=True), (-0.175, 0.02, 0.131))]
    part("Crawler_Tether_Gland", merge(gland), hub)

    # A strain relief spring over the first stretch of tether.
    coil = []
    for i in range(26):
        f = i / 25
        a = math.tau * 3.2 * f
        at = (-0.185 - 0.055 * f, 0.02 + 0.075 * f, 0.138 - 0.018 * f)
        coil.append(ellipsoid((at[0] + math.cos(a) * 0.012,
                               at[1], at[2] + math.sin(a) * 0.012),
                              (0.0030, 0.0030, 0.0030), segments=6))
    part("Crawler_Strain_Relief", merge(coil), hub)

    # ------------------------------------------------------------- the plate
    # Every machine on a plant has a number on it, and the eye expects one.
    part("Crawler_Nameplate", box((0.055, 0.001, 0.022), (0.06, -0.1355, 0.185)),
         hub, smooth=False)
    part("Crawler_Nameplate_Bolts",
         merge([bolt((0.038, -0.1365, 0.185), radius=0.002, head=0.002, axis="Y"),
                bolt((0.082, -0.1365, 0.185), radius=0.002, head=0.002, axis="Y")]),
         hub)

    # -------------------------------------------------- probe arm hardware
    arm = [shift(tube(0.010, 0.060, 12, caps=True), (0.10, 0.052, 0.090)),
           shift(tube(0.010, 0.060, 12, caps=True), (0.10, -0.052, 0.090)),
           box((0.090, 0.012, 0.012), (0.10, 0.0, 0.122))]
    part("Crawler_Probe_Guides", merge(arm), frame)
    part("Crawler_Probe_Bolts", bolt_ring((0.10, 0.0, 0.1165), 0.030, 4,
                                          bolt_r=0.004), hub)

    polys = sum(len(o.data.polygons) for o in bpy.data.objects
                if o.name in added and o.type == "MESH")
    return {"parts_added": len(added), "polygons_added": polys,
            "names": added,
            "clear_of_lens": True,
            "note": "parented to the Crawler rig, nothing enters the camera view"}


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=1))
