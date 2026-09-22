"""The bench rig: the same inspection loop, on a plate of real steel, from the parts list.

Run inside Blender:

    python scripts/blender_send.py tank/sim/blender_bench.py

Why this exists. The crawler in blender_tank.py is what the work scales to: a
magnetic machine that goes into a tank so a person does not have to. It is not
a thing that can be built this month, and a judge who only ever sees a
simulation is right to ask what is real. This is the machine that can be built
this month, modelled part for part from the bill of materials below, down to
the board it carries and the cells that power it. Same software, same prompts,
same scoring; a flat plate instead of a curved shell, and a hand-held gauge
instead of a probe on a couplant pump.

The plate is the honest part. Three weld runs laid by a fabricator, with the
faults put in on purpose: a stretch left unwelded, a grinder notch along a toe,
drilled pores, a hole burnt through. And on the BACK of the plate, where no
camera can ever see them, two ground pockets. From above the plate is perfect.
Only the thickness gauge finds them, and that is the whole argument of the
project standing on a bench for the price of a plate of steel.
"""

from __future__ import annotations

import importlib
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import meshkit  # noqa: E402

importlib.reload(meshkit)
from meshkit import (  # noqa: E402
    bead_profile,
    bead_surface,
    box,
    camera,
    chamfered,
    ellipsoid,
    emissive,
    look_at,
    make,
    merge,
    plain,
    rounded_plate,
    shift,
    steel_material,
    sweep,
    tube,
)

ROOT = HERE.parent.parent
OUT = ROOT / "tank" / "data" / "bench_scene.json"
SEED = 11

# Everything on this list is a thing that can be bought, and everything modelled
# below is on this list. Sizes are the real ones, in millimetres, so the render
# is a drawing of the rig rather than a picture of an idea. Prices are Mumbai,
# September 2026; the gauge is the only item that is not a component shop part.
BOM = [
    {"part": "2WD chassis plate, 220 x 150 x 3 acrylic", "qty": 1,
     "object": "Chassis_Plate", "inr": "600 to 1500 for the kit",
     "where": "Lamington Road"},
    {"part": "TT geared motor, 65 x 22 x 20, with 65 mm wheel", "qty": 2,
     "object": "Motor_*, Wheel_*", "inr": "in the kit", "where": "Lamington Road"},
    {"part": "Ball caster, 32 mm", "qty": 1, "object": "Caster_*",
     "inr": "in the kit", "where": "Lamington Road"},
    {"part": "L298N motor driver, 43 x 43 x 27", "qty": 1, "object": "Driver_L298N_*",
     "inr": "150 to 300", "where": "Lamington Road"},
    {"part": "ESP32-CAM with OV2640, 40.5 x 27", "qty": 1, "object": "ESP32_CAM_*",
     "inr": "500 to 900", "where": "Lamington Road"},
    {"part": "18650 holder, two cells, 78 x 40 x 21", "qty": 1,
     "object": "Battery_Holder, Battery_Cells", "inr": "500 to 1000",
     "where": "Lamington Road"},
    {"part": "Rocker switch, 21 x 15", "qty": 1, "object": "Power_Switch",
     "inr": "20 to 50", "where": "Lamington Road"},
    {"part": "M3 nylon standoffs, 20 mm", "qty": 4, "object": "Standoff_*",
     "inr": "50 for a pack", "where": "Lamington Road"},
    {"part": "LED module for inspection light, 12 x 12", "qty": 2,
     "object": "LED_Module_*", "inr": "150 to 300 the pair",
     "where": "Lamington Road, and it was missing from the first list"},
    {"part": "Hook-up wire, switch and heatshrink", "qty": 1, "object": "Wire_*",
     "inr": "300", "where": "Lamington Road"},
    {"part": "Mild steel plate, 600 x 300 x 6, with three weld runs", "qty": 1,
     "object": "Target_Plate, Bead_*", "inr": "500 to 1500",
     "where": "any fabricator, not Lamington Road"},
    {"part": "TM-8812 ultrasonic thickness gauge with 5 MHz probe", "qty": 1,
     "object": "Gauge_*, Probe_*", "inr": "10000",
     "where": "instrument dealer or online, not Lamington Road"},
]

# Millimetres, because every one of these came off a product page.
PLATE = (0.600, 0.300, 0.006)
BEAD_R = 0.0035                  # a bead-on-plate run stands about 3.5 mm proud
BEAD_LEN = 0.500
BEAD_Y = (-0.090, 0.0, 0.090)
RIPPLE_PITCH = 0.006             # a hand-laid run, tighter than a machine weld
PLATE_TOP = PLATE[2]

CAM_STANDOFF = 0.075             # what the mast holds the ESP32-CAM at
CAM_LENS_MM = 28.0               # about 66 degrees, which is the OV2640 lens

ACCEPTANCE = {
    "undercut": {"limit_mm": 0.8, "measure": "depth below the plate surface"},
    "porosity": {"limit_count": 4, "measure": "pores in any 100 mm of bead"},
    "crack": {"limit_mm": 0.0, "measure": "any crack is rejectable"},
    "lack_of_fusion": {"limit_mm": 0.0, "measure": "any break in the bead"},
    "burn_through": {"limit_mm": 0.0, "measure": "any hole through the joint"},
    "thinning": {"limit_fraction": 0.60,
                 "measure": "remaining wall as a fraction of the original"},
}


def build_materials() -> dict:
    return {
        "bench": plain("bench_top", (0.20, 0.145, 0.085), 0.0, 0.82),
        "plate": steel_material("bench_plate", (0.150, 0.158, 0.172), 0.58, 0.85,
                                48.0, 0.14, fine=900.0),
        "bead": steel_material("bench_bead", (0.175, 0.165, 0.140), 0.64, 0.60,
                               260.0, 0.20, fine=1400.0),
        "dark": plain("bench_dark", (0.016, 0.016, 0.020), 0.10, 0.88),
        "rust": steel_material("bench_rust", (0.27, 0.11, 0.035), 0.92, 0.10,
                               120.0, 0.9),
        "acrylic": plain("chassis_acrylic", (0.05, 0.09, 0.16), 0.0, 0.22),
        "motor": plain("motor_case", (0.72, 0.62, 0.10), 0.0, 0.45),
        "gearbox": plain("gearbox_case", (0.06, 0.06, 0.07), 0.0, 0.50),
        "tyre": plain("wheel_tyre", (0.035, 0.035, 0.038), 0.0, 0.80),
        "rim": plain("wheel_rim", (0.78, 0.62, 0.06), 0.0, 0.40),
        "pcb_red": plain("pcb_l298n", (0.42, 0.045, 0.045), 0.0, 0.52),
        "pcb_black": plain("pcb_esp32", (0.045, 0.048, 0.055), 0.0, 0.55),
        "heatsink": plain("heatsink", (0.10, 0.105, 0.115), 0.85, 0.42),
        "terminal": plain("terminal_block", (0.06, 0.14, 0.42), 0.0, 0.48),
        "cell": plain("cell_18650", (0.10, 0.28, 0.16), 0.20, 0.42),
        "holder": plain("cell_holder", (0.03, 0.03, 0.035), 0.0, 0.60),
        "brass": plain("standoff", (0.62, 0.50, 0.20), 0.90, 0.32),
        "steel_bright": plain("bracket", (0.55, 0.57, 0.60), 0.92, 0.35),
        "lens": plain("lens_glass", (0.03, 0.05, 0.08), 0.12, 0.06),
        "led": emissive("led_face", (1.0, 0.97, 0.90), 9.0),
        "wire_red": plain("wire_red", (0.40, 0.04, 0.04), 0.0, 0.55),
        "wire_black": plain("wire_black", (0.03, 0.03, 0.035), 0.0, 0.55),
        "gauge": plain("gauge_case", (0.62, 0.52, 0.06), 0.0, 0.50),
        "lcd": plain("gauge_lcd", (0.30, 0.38, 0.30), 0.0, 0.25),
        "probe": plain("ut_probe_case", (0.16, 0.17, 0.19), 0.55, 0.38),
    }


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


# ---------------------------------------------------------------- the target


def build_bench(mats, coll):
    make("Bench_Top", *box((1.30, 0.75, 0.025), (0.0, 0.0, -0.0125)),
         mat=mats["bench"], smooth=False, collection=coll)
    make("Target_Plate", *box(PLATE, (0.0, 0.0, PLATE[2] / 2)), mat=mats["plate"],
         smooth=False, collection=coll)


def build_beads(mats, coll, rng, defects):
    """Three weld runs, with the unwelded stretch left where it was planned."""
    gaps = {d["bead"]: (d["x_from_m"], d["x_to_m"]) for d in defects
            if d["kind"] == "lack_of_fusion"}
    for index, y in enumerate(BEAD_Y):
        name = "ABC"[index]
        spans = [(-BEAD_LEN / 2, BEAD_LEN / 2)]
        if name in gaps:
            lo, hi = gaps[name]
            spans = [(-BEAD_LEN / 2, lo), (hi, BEAD_LEN / 2)]
        parts = []
        for x0, x1 in spans:
            if x1 - x0 < 0.01:
                continue

            def place(f, across, out, x0=x0, x1=x1, y=y):
                return (x0 + (x1 - x0) * f, y + across, PLATE_TOP + out)

            parts.append(bead_surface(
                place, x1 - x0, bead_profile(rng, x1 - x0, RIPPLE_PITCH),
                BEAD_R, RIPPLE_PITCH, amp=0.10))
        make(f"Bead_{name}", *merge(parts), mat=mats["bead"], collection=coll)


def plan_defects() -> list[dict]:
    """The faults to put in the plate, written down before anything looks at it.

    Chosen so a fabricator can make every one of them with a torch and a
    grinder, and so that two of them are invisible from the side the camera is
    on. Positions are from the centre of the plate, x along the weld runs.
    """
    out = [
        {"id": "B001", "kind": "lack_of_fusion", "bead": "A", "y_m": BEAD_Y[0],
         "x_m": 0.050, "x_from_m": 0.020, "x_to_m": 0.080, "length_mm": 60.0,
         "how": "stop the run for 60 mm and restart", "visible": True,
         "rejectable": True},
        {"id": "B002", "kind": "undercut", "bead": "B", "y_m": BEAD_Y[1],
         "x_m": -0.110, "x_from_m": -0.150, "x_to_m": -0.070, "length_mm": 80.0,
         "depth_mm": 1.5, "how": "run a grinder along the toe of the bead",
         "visible": True, "rejectable": True},
        {"id": "B003", "kind": "porosity", "bead": "B", "y_m": BEAD_Y[1],
         "x_m": 0.100, "count": 5, "pore_mm": 2.0,
         "how": "drill five 2 mm dimples in the bead face", "visible": True,
         "rejectable": True},
        {"id": "B004", "kind": "burn_through", "bead": "C", "y_m": BEAD_Y[2],
         "x_m": -0.020, "hole_mm": 6.0,
         "how": "burn or drill a 6 mm hole through the bead", "visible": True,
         "rejectable": True},
        {"id": "B005", "kind": "thinning", "bead": None, "y_m": -0.040,
         "x_m": 0.200, "patch_mm": 40.0, "loss_mm": 2.6, "nominal_mm": 6.0,
         "remaining_mm": 3.4, "remaining_fraction": round(3.4 / 6.0, 3),
         "how": "grind a 40 mm pocket 2.6 mm deep in the BACK of the plate",
         "visible": False, "rejectable": True},
        {"id": "B006", "kind": "thinning", "bead": None, "y_m": 0.070,
         "x_m": -0.210, "patch_mm": 40.0, "loss_mm": 2.0, "nominal_mm": 6.0,
         "remaining_mm": 4.0, "remaining_fraction": round(4.0 / 6.0, 3),
         "how": "grind a 40 mm pocket 2.0 mm deep in the BACK of the plate",
         "visible": False, "rejectable": False},
        {"id": "B007", "kind": "pitting", "bead": None, "y_m": 0.115,
         "x_m": 0.235, "patch_mm": 55.0,
         "how": "leave a corner to rust, or etch it", "visible": True,
         "rejectable": False},
    ]
    for d in out:
        d["acceptance"] = ACCEPTANCE.get(d["kind"], {"measure": "reported, not gated"})
    return out


def build_defects(mats, coll, defects, rng):
    for d in defects:
        kind = d["kind"]
        if kind == "lack_of_fusion":
            # The bead simply is not there; what shows is the bare joint line.
            make(f"Defect_{d['id']}_bare_joint",
                 *box((d["x_to_m"] - d["x_from_m"], 0.0022, 0.0006),
                      (d["x_m"], d["y_m"], PLATE_TOP + 0.0003)),
                 mat=mats["dark"], smooth=False, collection=coll)

        elif kind == "undercut":
            depth = d["depth_mm"] / 1000
            make(f"Defect_{d['id']}_undercut",
                 *box((d["x_to_m"] - d["x_from_m"], 0.0035, depth * 1.4),
                      (d["x_m"], d["y_m"] + BEAD_R * 1.5, PLATE_TOP - depth * 0.3)),
                 mat=mats["dark"], smooth=False, collection=coll)

        elif kind == "porosity":
            pores = []
            for k in range(d["count"]):
                r = d["pore_mm"] / 2000 * rng.uniform(0.8, 1.2)
                x = d["x_m"] - 0.020 + 0.010 * k + rng.uniform(-0.002, 0.002)
                pores.append(ellipsoid((x, d["y_m"] + rng.uniform(-0.002, 0.002),
                                        PLATE_TOP + BEAD_R * 0.75),
                                       (r, r, r * 0.9), segments=10))
            make(f"Defect_{d['id']}_porosity", *merge(pores), mat=mats["dark"],
                 collection=coll)

        elif kind == "burn_through":
            r = d["hole_mm"] / 2000
            make(f"Defect_{d['id']}_hole",
                 *shift(tube(r, 0.010, 18, caps=True),
                        (d["x_m"], d["y_m"], PLATE_TOP - 0.002)),
                 mat=mats["dark"], collection=coll)

        elif kind == "thinning":
            # A pocket ground into the BACK face. From above there is nothing to
            # see, which is the point: only the gauge finds it.
            r = d["patch_mm"] / 2000
            make(f"Defect_{d['id']}_back_pocket",
                 *shift(tube(r, d["loss_mm"] / 1000, 28, caps=True),
                        (d["x_m"], d["y_m"], d["loss_mm"] / 2000)),
                 mat=mats["dark"], collection=coll)

        elif kind == "pitting":
            r = d["patch_mm"] / 2000
            skin, faces, segs = [], [], 32
            for i in range(segs + 1):
                for j in range(segs + 1):
                    a = math.tau * i / segs
                    reach = 0.72 + 0.18 * math.sin(3 * a) + 0.10 * math.sin(7 * a)
                    rr = r * reach * (j / segs)
                    skin.append((d["x_m"] + math.cos(a) * rr,
                                 d["y_m"] + math.sin(a) * rr, PLATE_TOP + 0.0004))
            for i in range(segs):
                for j in range(segs):
                    base = i * (segs + 1) + j
                    faces.append((base, base + 1, base + segs + 2, base + segs + 1))
            make(f"Defect_{d['id']}_rust", skin, faces, mats["rust"], collection=coll)
            pits = []
            for _ in range(rng.randint(22, 40)):
                a = rng.uniform(0, math.tau)
                rr = r * 0.8 * math.sqrt(rng.uniform(0, 1))
                pr = rng.uniform(0.0008, 0.0022)
                pits.append(ellipsoid((d["x_m"] + math.cos(a) * rr,
                                       d["y_m"] + math.sin(a) * rr, PLATE_TOP + 0.0006),
                                      (pr, pr, pr * 0.5), segments=8))
            make(f"Defect_{d['id']}_pits", *merge(pits), mat=mats["dark"],
                 collection=coll)


# ------------------------------------------------------------------ the robot


def build_robot(mats, coll, x_centre):
    """Every part on the list, at the size it arrives in the post.

    The frame is the robot's own: +X is the way it drives, +Z is up off the
    plate, and the whole thing is carried onto the plate by one matrix, so the
    same code parks it anywhere along a weld run.
    """
    rig = bpy.data.objects.new("Bench_Rig", None)
    rig.empty_display_size = 0.05
    coll.objects.link(rig)

    def part(name, verts, faces, mat, smooth=True):
        obj = make(name, verts, faces, mat, smooth=smooth, collection=coll)
        obj.parent = rig
        return obj

    deck_z = 0.0415
    # 220 x 150 x 3 acrylic, the plate the kit is built around.
    part("Chassis_Plate", *rounded_plate((0.220, 0.150, 0.003),
                                         (0.0, 0.0, deck_z), radius=0.008),
         mat=mats["acrylic"], smooth=False)

    # Two TT gearmotors under the deck, wheels outboard of it.
    for side, sy in (("L", 0.045), ("R", -0.045)):
        part(f"Motor_{side}_Gearbox", *box((0.065, 0.022, 0.020),
                                           (-0.055, sy, deck_z - 0.012)),
             mat=mats["gearbox"], smooth=False)
        part(f"Motor_{side}_Can", *shift(tube(0.0115, 0.025, 16, caps=True),
                                         (-0.020, sy, deck_z - 0.012)),
             mat=mats["motor"])
    spin = Matrix.Rotation(math.pi / 2, 4, "X")
    tyres, rims = [], []
    for sy in (-0.088, 0.088):
        place = Matrix.Translation((-0.055, sy, 0.0325)) @ spin
        for src, sink in ((tube(0.0325, 0.026, 26, caps=True), tyres),
                          (tube(0.020, 0.028, 20, caps=True), rims)):
            v, f = src
            sink.append(([tuple(place @ Vector(p)) for p in v], f))
    part("Wheel_Tyres", *merge(tyres), mat=mats["tyre"])
    part("Wheel_Rims", *merge(rims), mat=mats["rim"])

    # One ball caster at the front, which is what carries the third corner.
    part("Caster_Housing", *box((0.032, 0.032, 0.016), (0.085, 0.0, deck_z - 0.010)),
         mat=mats["steel_bright"], smooth=False)
    part("Caster_Ball", *ellipsoid((0.085, 0.0, 0.0125), (0.0125, 0.0125, 0.0125),
                                   segments=14), mat=mats["steel_bright"])

    # L298N on four standoffs, heatsink and screw terminals and all.
    board_z = deck_z + 0.0015 + 0.020
    part("Driver_L298N_PCB", *box((0.043, 0.043, 0.0016), (-0.005, 0.0, board_z)),
         mat=mats["pcb_red"], smooth=False)
    part("Driver_L298N_Heatsink", *box((0.023, 0.023, 0.012),
                                       (-0.005, 0.0, board_z + 0.0068)),
         mat=mats["heatsink"], smooth=False)
    for k, sx in enumerate((-0.023, 0.013)):
        part(f"Driver_L298N_Terminal_{k}", *box((0.010, 0.030, 0.010),
                                                (-0.005 + sx, 0.0, board_z + 0.0058)),
             mat=mats["terminal"], smooth=False)
    posts = []
    for sx in (-0.023, 0.013):
        for sy in (-0.018, 0.018):
            posts.append(shift(tube(0.0030, 0.020, 6, caps=True),
                               (-0.005 + sx, sy, deck_z + 0.0115)))
    part("Standoff_L298N", *merge(posts), mat=mats["brass"], smooth=False)

    # Two 18650 cells in a holder at the back, where the weight belongs.
    part("Battery_Holder", *box((0.078, 0.040, 0.021), (-0.072, 0.0, deck_z + 0.0120)),
         mat=mats["holder"], smooth=False)
    cells = []
    for sy in (-0.0095, 0.0095):
        cells.append(shift(tube(0.0092, 0.065, 18, caps=True),
                           (-0.072, sy, deck_z + 0.0150)))
    part("Battery_Cells", *merge(cells), mat=mats["cell"])
    part("Power_Switch", *box((0.021, 0.015, 0.013), (-0.100, 0.055, deck_z + 0.008)),
         mat=mats["dark"], smooth=False)

    # The mast, and the ESP32-CAM on the end of it looking straight down. It has
    # to reach past the chassis or the board photographs its own robot.
    part("Cam_Mast_Post", *box((0.012, 0.012, 0.100), (0.100, 0.0, deck_z + 0.0515)),
         mat=mats["steel_bright"], smooth=False)
    arm_z = PLATE_TOP + CAM_STANDOFF
    part("Cam_Mast_Arm", *box((0.100, 0.014, 0.006), (0.148, 0.0, arm_z + 0.006)),
         mat=mats["steel_bright"], smooth=False)
    part("ESP32_CAM_PCB", *box((0.0405, 0.027, 0.0012), (0.185, 0.0, arm_z + 0.0015)),
         mat=mats["pcb_black"], smooth=False)
    part("ESP32_CAM_Module", *box((0.0085, 0.0085, 0.006), (0.185, 0.0, arm_z - 0.0021)),
         mat=mats["dark"], smooth=False)
    part("ESP32_CAM_Lens", *shift(tube(0.0035, 0.003, 14, caps=True),
                                  (0.185, 0.0, arm_z - 0.0062)), mat=mats["lens"])
    part("ESP32_CAM_Antenna", *box((0.015, 0.003, 0.0012),
                                   (0.163, 0.010, arm_z + 0.0028)),
         mat=mats["steel_bright"], smooth=False)

    # The two LED modules the first parts list forgot. A weld is read by the
    # shadow it throws, so they sit wide and shine in across the bead.
    for sy in (-0.045, 0.045):
        part(f"LED_Module_PCB_{sy:+.3f}", *box((0.012, 0.012, 0.002),
                                               (0.185, sy, arm_z - 0.001)),
             mat=mats["pcb_black"], smooth=False)
        part(f"LED_Module_Dome_{sy:+.3f}",
             *ellipsoid((0.185, sy, arm_z - 0.004), (0.004, 0.004, 0.003), segments=10),
             mat=mats["led"])
    part("LED_Bracket", *box((0.014, 0.110, 0.003), (0.185, 0.0, arm_z + 0.0035)),
         mat=mats["steel_bright"], smooth=False)

    # Wiring, because a rig with no wires on it is a drawing of a rig.
    part("Wire_Battery_Red",
         *sweep([(-0.072, 0.020, deck_z + 0.021), (-0.090, 0.045, deck_z + 0.016),
                 (-0.100, 0.055, deck_z + 0.012)], 0.0013, 6), mat=mats["wire_red"])
    part("Wire_Switch_Driver",
         *sweep([(-0.100, 0.055, deck_z + 0.012), (-0.060, 0.035, deck_z + 0.018),
                 (-0.018, 0.018, board_z + 0.004)], 0.0013, 6), mat=mats["wire_red"])
    part("Wire_Driver_Motor_L",
         *sweep([(0.008, 0.018, board_z + 0.004), (0.000, 0.040, deck_z + 0.006),
                 (-0.022, 0.045, deck_z - 0.008)], 0.0013, 6), mat=mats["wire_black"])
    part("Wire_Driver_Motor_R",
         *sweep([(0.008, -0.018, board_z + 0.004), (0.000, -0.040, deck_z + 0.006),
                 (-0.022, -0.045, deck_z - 0.008)], 0.0013, 6), mat=mats["wire_black"])
    # The camera lead is cable-tied to the mast and the arm, the way anybody
    # would run it. Left to fly straight to the board it arced over the whole
    # rig like a handle.
    part("Wire_Cam_Power",
         *sweep([(-0.005, -0.020, board_z + 0.002), (0.040, -0.026, deck_z + 0.009),
                 (0.088, -0.010, deck_z + 0.008), (0.093, -0.008, deck_z + 0.045),
                 (0.096, -0.008, arm_z + 0.004), (0.140, -0.009, arm_z + 0.009),
                 (0.178, -0.008, arm_z + 0.005)], 0.0011, 6), mat=mats["wire_red"])

    # The OV2640 focuses close, and so must the render camera.
    # The OV2640 on a 2.8 mm lens is about 66 degrees, which is 28 mm on a full
    # frame. Held 120 mm off the plate it covers 154 by 87 mm, so a 500 mm weld
    # run is four frames and a 2 mm pore is twenty pixels.
    cam = camera("ESP32_Cam", CAM_LENS_MM, coll)
    cam.parent = rig
    # Below the lens barrel, not level with it, or the camera photographs the
    # inside of its own housing.
    cam.matrix_basis = look_at((0.185, 0.0, arm_z - 0.010), (0.185, 0.0, PLATE_TOP),
                               up_hint=(0.0, 1.0, 0.0))

    for sy in (-0.045, 0.045):
        lamp = bpy.data.lights.new(f"LED_{sy:+.3f}", type="SPOT")
        lamp.energy = 0.17
        lamp.spot_size = math.radians(100)
        lamp.spot_blend = 0.6
        lamp.shadow_soft_size = 0.004
        obj = bpy.data.objects.new(lamp.name, lamp)
        obj.parent = rig
        coll.objects.link(obj)
        obj.matrix_basis = look_at((0.185, sy, arm_z - 0.005),
                                   (0.185, sy * 0.15, PLATE_TOP), up_hint=(1.0, 0.0, 0.0))

    rig.matrix_world = Matrix.Translation((x_centre, BEAD_Y[1], PLATE_TOP))
    return rig, cam


def build_gauge(mats, coll):
    """The TM-8812 and its probe, lying on the bench where a person left them."""
    x, y = 0.055, -0.245
    make("Gauge_Body", *chamfered((0.120, 0.062, 0.030), (x, y, 0.015), 0.004),
         mat=mats["gauge"], smooth=False, collection=coll)
    make("Gauge_LCD", *box((0.045, 0.025, 0.002), (x + 0.028, y, 0.0305)),
         mat=mats["lcd"], smooth=False, collection=coll)
    keys = []
    for k in range(4):
        keys.append(box((0.012, 0.012, 0.003), (x - 0.030 + 0.016 * (k % 2),
                                                y - 0.012 + 0.024 * (k // 2), 0.031)))
    make("Gauge_Keys", *merge(keys), mat=mats["dark"], smooth=False, collection=coll)

    px, py = -0.115, -0.238
    make("Probe_Body", *shift(tube(0.009, 0.025, 18, caps=True), (px, py, 0.0125)),
         mat=mats["probe"], collection=coll)
    make("Probe_Face", *shift(tube(0.0085, 0.002, 18, caps=True), (px, py, 0.0008)),
         mat=mats["dark"], collection=coll)
    # A coiled lead, because that is how the probe comes and it reads instantly.
    coil = []
    turns, steps = 6, 90
    for i in range(steps + 1):
        f = i / steps
        a = math.tau * turns * f
        coil.append((px + (x - 0.060 - px) * f + 0.010 * math.cos(a),
                     py + (y - py) * f * 0.35 + 0.010 * math.sin(a),
                     0.006 + 0.004 * math.sin(a)))
    make("Probe_Cable", *sweep(coil, 0.0018, 6), mat=mats["dark"], collection=coll)


# ------------------------------------------------------------------- the shot


def build_lighting(coll):
    """A room with a bench in it, not a tank. The rig still carries its own light."""
    key = bpy.data.lights.new("Room_Key", type="AREA")
    key.energy = 70.0
    key.size = 1.4
    obj = bpy.data.objects.new("Room_Key", key)
    obj.matrix_world = look_at((0.45, -0.75, 1.35), (0.0, 0.0, 0.0))
    coll.objects.link(obj)

    fill = bpy.data.lights.new("Room_Fill", type="AREA")
    fill.energy = 28.0
    fill.size = 2.0
    obj = bpy.data.objects.new("Room_Fill", fill)
    obj.matrix_world = look_at((-0.9, 0.85, 1.0), (0.0, 0.0, 0.0))
    coll.objects.link(obj)


def build_cameras(coll, rig):
    cam = camera("Bench_Overview", 42.0, coll)
    cam.matrix_world = look_at((-0.52, -0.62, 0.46), (0.03, -0.02, 0.03))

    detail = camera("Bench_Detail", 60.0, coll)
    detail.matrix_world = rig.matrix_world @ look_at((-0.26, -0.40, 0.29),
                                                     (0.11, 0.0, 0.035))
    return cam, detail


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
    for attr, value in (("taa_render_samples", 96), ("use_raytracing", True)):
        try:
            setattr(scene.eevee, attr, value)
        except (AttributeError, TypeError):
            continue
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
            node.inputs[0].default_value = (0.055, 0.058, 0.065, 1.0)


def main() -> dict:
    rng = random.Random(SEED)
    clear_scene()
    mats = build_materials()
    stage = collection("Bench")
    faults = collection("Plate_Defects")
    robot = collection("Rig")

    defects = plan_defects()
    build_bench(mats, stage)
    build_beads(mats, stage, rng, defects)
    build_defects(mats, faults, defects, rng)

    # Parked so the camera on the mast is over the drilled pores, which is a
    # fault the ground truth already has an answer for.
    subject = next(d for d in defects if d["kind"] == "porosity")
    rig, cam = build_robot(mats, robot, subject["x_m"] - 0.185)
    build_gauge(mats, stage)
    build_lighting(stage)
    overview, detail = build_cameras(robot, rig)
    set_render(bpy.context.scene, overview)

    ground_truth = {
        "units": "metres unless a key says mm",
        "seed": SEED,
        "what_this_is": (
            "The bench rig: the same inspection loop on a plate of real steel, "
            "built from parts that can be bought. The tank scene is what it "
            "scales to."
        ),
        "bill_of_materials": BOM,
        "plate": {"size_m": list(PLATE), "thickness_mm": PLATE[2] * 1000,
                  "bead_runs": len(BEAD_Y), "bead_length_m": BEAD_LEN,
                  "bead_y_m": list(BEAD_Y), "bead_proud_mm": BEAD_R * 1000},
        "acceptance": ACCEPTANCE,
        "defects": defects,
        "camera": {"part": "ESP32-CAM, OV2640", "lens_mm": CAM_LENS_MM,
                   "standoff_m": CAM_STANDOFF,
                   "covers_mm": [round(36.0 / CAM_LENS_MM * CAM_STANDOFF * 1000, 1),
                                 round(20.25 / CAM_LENS_MM * CAM_STANDOFF * 1000, 1)]},
        "parked_over": subject["id"],
        "thickness_model": (
            "nominal is 6.0 mm everywhere. A thinning defect removes loss_mm inside "
            "patch_mm of its centre, ground into the back face, so nothing on the "
            "camera side changes and only the gauge finds it."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ground_truth, indent=1), encoding="utf-8")

    named = {o.name for o in bpy.data.objects}
    return {
        "objects": len(bpy.data.objects),
        "polygons": sum(len(o.data.polygons) for o in bpy.data.objects
                        if getattr(o, "data", None) and hasattr(o.data, "polygons")),
        "bom_lines": len(BOM),
        "bom_missing": [item["part"] for item in BOM
                        if not all(any(n.startswith(prefix.strip().replace("*", ""))
                                       for n in named)
                                   for prefix in item["object"].split(","))],
        "defects": {d["kind"]: sum(1 for x in defects if x["kind"] == d["kind"])
                    for d in defects},
        "invisible_to_the_camera": sum(1 for d in defects if not d["visible"]),
        "parked_over": subject["id"],
        "ground_truth": str(OUT),
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
