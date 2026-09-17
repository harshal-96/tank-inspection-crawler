"""Build the welding cell in Blender, in the cell's own coordinates.

This is the same cell the simulation runs, not a picture of something like it.
The table, the locating pads and the part are placed from cell.geometry, and the
part's panels are bent by cell.forming, so the seam in the render is in the same
place the seam tracker measures.

Run it through the running Blender:

    python scripts/blender_send.py code scripts/build_cell_blender.py
"""

import math
import random
import sys

sys.path.insert(0, r"D:\hackathon\nvidia_nebius")

import bpy  # noqa: E402
from mathutils import Euler, Vector  # noqa: E402

from cell import forming as forming_lib  # noqa: E402
from cell import geometry as geo  # noqa: E402

SEED = 11
FORMING_SCALE = 1.0

# Where the camera stands while the arc is running, and where it looks.
#
# cell.geometry holds a different pose, and both are right. That one is the
# pre-weld scan: the part arrives, the cell photographs it, then it is welded, so
# there is no torch in the way and the camera can sit square on the joint at the
# height it is welded at.
#
# This one is for watching while welding, and it has to solve a harder problem.
# The torch is held out from the corner along the outward diagonal because that
# is the work angle the joint needs, and the camera wants that same diagonal
# because it is the only direction that opens both folded faces equally, which is
# what fitting a line to each face and intersecting them requires. Two
# requirements, one piece of air.
#
# Raising the camera and aiming it at the seam ABOVE the arc resolves it. It then
# looks over the torch rather than through it, at metal that has not been welded
# yet, which is the only metal a correction can still be applied to. Aiming at the
# torch's own height instead was what made every robot base position fail: the
# blocker was the torch, not the arm.
#
# Both numbers were found by firing rays at the seam from candidate poses and
# counting what arrived, not by moving the camera until the picture looked right.
LOOKAHEAD_EYE = (5.308, -1.125, 1.439)
LOOKAHEAD_AIM_Z = 1.0995

# Thirty degrees off the corner's diagonal rather than on it, which swings the
# camera round to the side the arm is not working from. It is a trade, not a free
# move: on the diagonal both folded faces sit at forty five degrees and the line
# fit gets equal evidence from each, while here the near face opens to fifty five
# and the far one closes to twenty eight. Twenty eight degrees is still enough
# metal to fit a line to. Going further round, to twenty degrees off, was tested
# and reaches nineteen, which starts starving the fit on that face for nothing in
# return.

# The base had to move too. On the diagonal it stood exactly where the camera
# needs to see, so it sits off to the front of the table instead. From here the
# arm still reaches the torch pose, to within about a millimetre and a half.
ROBOT_BASE = (3.84, -2.22)

# The corner the robot welds, in cell coordinates.
JOINT_X = geo.FIXTURE_X + geo.JOINT_LOCAL[0]
JOINT_Y = geo.JOINT_LOCAL[1]
SEAM_BOTTOM = geo.TABLE_TOP + geo.PART_T
SEAM_TOP = SEAM_BOTTOM + geo.PART_D


# ------------------------------------------------------------------- materials


def material(name, base, metallic=0.0, roughness=0.5, emission=None):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        for key in ("Emission Color", "Emission"):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = (*emission[0], 1.0)
                break
        bsdf.inputs["Emission Strength"].default_value = emission[1]
    return mat


def build_materials():
    return {
        # Mill finish stainless: bright, but rolled rather than polished, so it is
        # rough enough to hold a diffuse highlight instead of mirroring the room.
        "sheet": material("sheet_steel", (0.560, 0.574, 0.588), 1.0, 0.28),
        "table_top": material("table_top", (0.210, 0.226, 0.244), 0.85, 0.42),
        "table_frame": material("table_frame", (0.112, 0.125, 0.140), 0.6, 0.55),
        # Locating pads. Hardened tool steel, darkened by use.
        "pad": material("locating_pad", (0.150, 0.100, 0.080), 0.7, 0.40),
        "belt": material("belt", (0.045, 0.048, 0.052), 0.0, 0.85),
        "roller": material("roller", (0.400, 0.410, 0.420), 0.9, 0.30),
        "frame": material("conveyor_frame", (0.520, 0.540, 0.560), 0.85, 0.35),
        "robot": material("robot_body", (0.620, 0.632, 0.646), 0.15, 0.38),
        "robot_dark": material("robot_joint", (0.120, 0.130, 0.145), 0.5, 0.45),
        "robot_trim": material("robot_trim", (0.760, 0.300, 0.060), 0.1, 0.45),
        "torch": material("torch", (0.070, 0.072, 0.078), 0.4, 0.35),
        "nozzle": material("nozzle", (0.680, 0.400, 0.220), 1.0, 0.25),
        "sensor": material("sensor_head", (0.090, 0.095, 0.105), 0.3, 0.30),
        "lens": material("sensor_window", (0.02, 0.05, 0.09), 0.0, 0.10),
        # Weld metal: oxidised, so duller and darker than the sheet beside it.
        "bead": material("weld_bead", (0.235, 0.220, 0.205), 0.8, 0.62),
        "bead_hot": material("weld_bead_hot", (0.90, 0.42, 0.10), 0.3, 0.55,
                             emission=((1.0, 0.52, 0.14), 7.0)),
        "floor": material("floor", (0.185, 0.190, 0.198), 0.0, 0.75),
        "guard": material("guard", (0.820, 0.640, 0.060), 0.2, 0.50),
    }


# -------------------------------------------------------------------- geometry


def put(obj, mat, collection="CELL"):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.name = f"{collection}_{obj.name}"
    return obj


def slab(name, x0, x1, y0, y1, z0, z1, mat):
    mesh = bpy.data.meshes.new(name)
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return put(obj, mat)


def tube(name, start, end, radius, mat, faces=24):
    start, end = Vector(start), Vector(end)
    axis = end - start
    length = axis.length
    bpy.ops.mesh.primitive_cylinder_add(vertices=faces, radius=radius, depth=length)
    obj = bpy.context.active_object
    obj.name = name
    obj.location = (start + end) / 2.0
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = axis.to_track_quat("Z", "Y")
    return put(obj, mat)


def beam(name, start, end, width, height, mat):
    """A boxy link between two points. Arms are fabricated castings, not pipes."""
    start, end = Vector(start), Vector(end)
    axis = end - start
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (width, height, axis.length)
    obj.location = (start + end) / 2.0
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = axis.to_track_quat("Z", "Y")
    return put(obj, mat)


def bent_side(name, axis, inner, outer, span_lo, span_hi, panel, outward, mat, segments=40):
    """A folded side of the tray, leaning and bowing the way the press left it.

    The same construction the simulation uses, so the corner in this picture is
    where the corner in the measurement is.
    """
    wall = geo.PART_T + geo.PART_D

    def ring(z):
        off = outward * panel.offset_m(max(z - geo.PART_T, 0.0), geo.PART_D)
        near, far = inner + off, outer + off
        corners = ([(near, span_lo), (far, span_lo), (far, span_hi), (near, span_hi)]
                   if axis == 0 else
                   [(span_lo, near), (span_lo, far), (span_hi, far), (span_hi, near)])
        area = sum(corners[i][0] * corners[(i + 1) % 4][1]
                   - corners[(i + 1) % 4][0] * corners[i][1] for i in range(4))
        if area < 0:
            corners.reverse()
        return [(c[0], c[1], z) for c in corners]

    rings = [ring(wall * i / segments) for i in range(segments + 1)]
    verts = [v for r in rings for v in r]
    faces = []
    for i in range(segments):
        lo, hi = i * 4, (i + 1) * 4
        for k in range(4):
            faces.append([lo + k, lo + (k + 1) % 4, hi + (k + 1) % 4, hi + k])
    faces.append([3, 2, 1, 0])
    top = segments * 4
    faces.append([top, top + 1, top + 2, top + 3])

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return put(obj, mat)


def build_part(mats, tray_forming, part_x, part_y):
    """The tray, standing on the fixture where it came to rest."""
    hw, hh, t = geo.PART_W / 2.0, geo.PART_H / 2.0, geo.PART_T
    z0 = geo.TABLE_TOP
    made = [slab("tray_base", -hw, hw, -hh, hh, 0.0, t, mats["sheet"])]
    for name, axis, inner, outer, lo, hi, outward, which in (
        ("tray_side_right", 0, hw, hw + t, -hh, hh, +1.0, "right"),
        ("tray_side_left", 0, -hw, -(hw + t), -hh, hh, -1.0, "left"),
        ("tray_end_back", 1, hh, hh + t, -hw, hw, +1.0, "back"),
        ("tray_end_front", 1, -hh, -(hh + t), -hw, hw, -1.0, "front"),
    ):
        made.append(bent_side(name, axis, inner, outer, lo, hi,
                              getattr(tray_forming, which), outward, mats["sheet"]))
    for obj in made:
        obj.location = (part_x, part_y, z0)
        for polygon in obj.data.polygons:
            polygon.use_smooth = False
    return made


def build_fixture(mats):
    """The table the part is welded on, and the pads that stop it."""
    x, top = geo.FIXTURE_X, geo.TABLE_TOP
    slab("table_top", x - 0.80, x + 0.80, -0.85, 0.85, top - 0.05, top, mats["table_top"])
    for corner_x in (x - 0.70, x + 0.70):
        for corner_y in (-0.72, 0.72):
            slab("table_leg", corner_x - 0.04, corner_x + 0.04,
                 corner_y - 0.04, corner_y + 0.04, 0.0, top - 0.05, mats["table_frame"])
    slab("table_rail", x - 0.74, x + 0.74, -0.76, -0.70, 0.20, 0.26, mats["table_frame"])
    slab("table_rail_b", x - 0.74, x + 0.74, 0.70, 0.76, 0.20, 0.26, mats["table_frame"])

    # The end stop, as two short pads set well inboard of the welded corner so
    # they locate the part without standing in the weld or in the camera's way.
    for side in (-1, 1):
        slab("locating_pad", x + 0.460, x + 0.510,
             side * 0.34 - 0.14, side * 0.34 + 0.14,
             top, top + 0.12, mats["pad"])


def build_conveyor(mats):
    """The line that feeds parts to the fixture."""
    top = geo.TABLE_TOP
    x0, x1 = -0.2, geo.FIXTURE_X - 0.82
    slab("belt", x0, x1, -0.62, 0.62, top - 0.03, top, mats["belt"])
    for side in (-1, 1):
        slab("conveyor_side", x0, x1, side * 0.66 - 0.03, side * 0.66 + 0.03,
             top - 0.16, top - 0.012, mats["frame"])
        # Low guide strips, just proud of the belt, to keep parts tracking.
        slab("conveyor_guide", x0, x1, side * 0.60 - 0.012, side * 0.60 + 0.012,
             top, top + 0.035, mats["frame"])
    steps = int((x1 - x0) / 0.24)
    for i in range(steps + 1):
        x = x0 + i * 0.24
        tube("roller", (x, -0.63, top - 0.115), (x, 0.63, top - 0.115), 0.030,
             mats["roller"])
    for i in range(4):
        x = x0 + 0.35 + i * (x1 - x0 - 0.7) / 3.0
        for side in (-1, 1):
            slab("conveyor_leg", x - 0.035, x + 0.035,
                 side * 0.66 - 0.035, side * 0.66 + 0.035, 0.0, top - 0.14,
                 mats["frame"])


# The arm's triangles arrive disconnected: every edge in every casting belongs to
# exactly one face, and all the shading is carried by per-triangle split normals.
# That is why the wrist shades in hard patches.
#
# Welding them back together was tried both ways and both are worse than leaving
# it alone. Re-deriving the shading from face angles gives castings that look
# inflated at a generous threshold and faceted at a tight one. Keeping the
# original split normals through the merge shatters the wrist outright, because
# that data was authored against the un-merged topology and no longer maps to it
# once vertices are shared.
#
# So the import is left as the file ships. The switch stays because the finding is
# worth keeping: the fix for this is a better source mesh, not a repair pass.
WELD_ARM_MESHES = False

KUKA_GLB = r"D:\hackathon\nvidia_nebius\assets\robot\kr16_2.glb"

# Which local axis each joint of this model actually turns about. The link frames
# in the file are not the ones the datasheet draws, so these were measured by
# turning each joint and watching which way the flange went, rather than assumed:
# turning about the wrong one moves the arm sideways through its own castings.
ARM_AXES = (("link_1", 1), ("link_2", 2), ("link_3", 2), ("link_5", 2))


# The paint the model was built with. The file lost its colours on export but kept
# the spec in the material names, so this is the machine's own livery read off the
# model rather than a guess at what shade of orange looks right. Values are linear,
# converted from the RAL sRGB hex.
RAL = {
    "RAL2003": ((0.946, 0.198, 0.032), 0.42),   # pastel orange, the arm castings
    "RAL9005": ((0.015, 0.015, 0.016), 0.34),   # jet black, base and motor covers
}


def paint_kuka():
    """Put the RAL colours back on, by the names the materials already carry."""
    painted = 0
    for mat in bpy.data.materials:
        for code, (colour, roughness) in RAL.items():
            if not mat.name.startswith(code):
                continue
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                continue
            bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = 0.0
            painted += 1
            break
    return painted


def weld_arm_meshes(threshold=0.0002):
    """Join the arm's triangles back together, keeping the shading it came with.

    The file arrives as a triangle soup: every edge in every casting belongs to
    exactly one face, which is to say no triangle shares an edge with its
    neighbour, because the exporter split every vertex to carry its own normal.
    Disconnected triangles crack along their joins and shade unpredictably where
    the split normals disagree, which is the patchy faceting on the wrist.

    Merging vertices that sit on top of each other rebuilds the connectivity.
    The custom split normals are deliberately left alone: they are the shading the
    model was authored with, and an earlier attempt that discarded them and
    re-derived smoothing from face angles produced castings that looked inflated
    at a generous angle and faceted at a tight one. Neither reproduced what the
    author drew, because no single threshold can.
    """
    import bmesh

    welded = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.name.startswith("CELL_"):
            continue
        mesh = obj.data
        before = len(mesh.vertices)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=threshold)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
        if len(mesh.vertices) < before:
            welded += 1
    return welded


def import_kuka(base, mats):
    """Bring in the KR16-2 and stand it on its plinth."""
    bpy.ops.import_scene.gltf(filepath=KUKA_GLB)
    print(f"[cell] painted {paint_kuka()} arm materials")
    if WELD_ARM_MESHES:
        print(f"[cell] welded {weld_arm_meshes()} arm meshes back together")
    root = bpy.data.objects["kuka_kr16_2"]
    root.location = Vector((base.x, base.y, 0.10))
    for name, _ in ARM_AXES:
        bpy.data.objects[name].rotation_mode = "XYZ"
        bpy.data.objects[name].rotation_euler = (0.0, 0.0, 0.0)
    slab("robot_plinth", base.x - 0.36, base.x + 0.36, base.y - 0.36, base.y + 0.36,
         0.0, 0.10, mats["robot_dark"])
    return root


def tool_flange_local():
    """Where the tool actually bolts on, in link 6's own frame.

    The origin of link 6 is not the mounting face: on this model it sits about
    150 mm back inside the wrist casting. Building a torch from the origin puts
    the torch inside the robot, and solving the arm against the origin aims the
    wrong point at the seam. So the face is measured off the mesh: take the axis
    the flange disc runs along, then average the ring of vertices furthest out
    along it.
    """
    link6 = bpy.data.objects["link_6"]
    mesh = bpy.data.objects["link_6.001"]
    inverse = link6.matrix_world.inverted()
    points = [inverse @ (mesh.matrix_world @ v.co) for v in mesh.data.vertices]
    if not points:
        return Vector((0.0, 0.0, 0.0))

    axis = max(points, key=lambda p: p.length).normalized()
    reach = max(p.dot(axis) for p in points)
    face = [p for p in points if p.dot(axis) > reach - 0.004]
    centre = Vector((0.0, 0.0, 0.0))
    for p in face:
        centre += p
    return centre / len(face)


def tool_world(local):
    bpy.context.view_layer.update()
    return bpy.data.objects["link_6"].matrix_world @ local


def solve_arm(target, tool_local, passes=7):
    """Point the arm at a target by asking Blender where it currently is.

    Rather than derive the inverse kinematics of a link frame layout that is not
    the one on the drawing, each joint is nudged and the tool is measured. It
    converges in a few passes and cannot be wrong about the model's own geometry,
    because the model is what is being measured.
    """
    joints = [(bpy.data.objects[name], axis) for name, axis in ARM_AXES]

    def miss():
        return (tool_world(tool_local) - target).length

    best = miss()
    for step in (0.6, 0.25, 0.1, 0.04, 0.015, 0.006, 0.002)[:passes]:
        improved = True
        while improved:
            improved = False
            for joint, axis in joints:
                for delta in (step, -step):
                    keep = list(joint.rotation_euler)
                    angles = list(keep)
                    angles[axis] += delta
                    joint.rotation_euler = angles
                    score = miss()
                    if score < best - 1e-6:
                        best, improved = score, True
                    else:
                        joint.rotation_euler = keep
    return best, tool_world(tool_local)


def build_robot(mats):
    """The KR16-2, posed with its torch on the seam and the sensor ahead of it.

    The arm is the real model rather than a stand-in, so its reach and proportions
    are the machine's own. Where it ends up is solved against the corner this cell
    actually welds, so if the part moves the arm moves with it.
    """
    base = Vector((ROBOT_BASE[0], ROBOT_BASE[1], 0.0))
    torch_z = (SEAM_BOTTOM + SEAM_TOP) / 2.0 - 0.02

    import_kuka(base, mats)
    tool_local = tool_flange_local()

    joint = Vector((JOINT_X, JOINT_Y, torch_z))
    out = Vector((geo.JOINT_LOCAL[0], geo.JOINT_LOCAL[1], 0.0)).normalized()
    along = Vector((0.0, 0.0, 1.0))               # the seam runs vertically
    across = out.cross(along).normalized()        # sideways, along the joint face

    # The wire tip stands off the corner by the cell's own stick-out. The nozzle
    # cup stops well short of it, because a cup pushed down to the work would sit
    # in the weld: on a real torch only the wire crosses that last stretch.
    wire_tip = joint + out * geo.STICKOUT_M
    nozzle_face = wire_tip + out * 0.014 + along * 0.004

    # Torch held out from the corner and dragged back up the seam, which is the
    # work angle a vertical joint is welded at.
    approach = (out * 0.86 + along * 0.50).normalized()
    mount = nozzle_face + approach * 0.235
    flange_target = mount + approach * 0.045

    miss, flange_at = solve_arm(flange_target, tool_local)
    print(f"[cell] arm solved to {miss * 1000:.1f} mm of its target")

    axis_dir = (nozzle_face - flange_at).normalized()
    neck = flange_at + axis_dir * 0.145
    # The bracket bolts to the flange face, so it sits against the wrist by
    # design. It is kept no wider than the flange itself, or it reads as a plate
    # sunk into the casting rather than a mount sitting on it.
    tube("torch_mount", flange_at, flange_at + axis_dir * 0.048, 0.032,
         mats["robot_dark"])
    tube("torch_body", flange_at + axis_dir * 0.035, neck, 0.029, mats["torch"])
    tube("torch_shroud", neck, nozzle_face - axis_dir * 0.035, 0.020, mats["torch"])
    tube("torch_nozzle", nozzle_face - axis_dir * 0.040, nozzle_face, 0.0125,
         mats["nozzle"])
    # Only the wire crosses the stand-off.
    tube("weld_wire", nozzle_face, wire_tip, 0.0007, mats["nozzle"], faces=8)

    # Cable pack looping off the back of the mount, clear of the wrist.
    tube("conduit", flange_at + axis_dir * 0.03 - across * 0.075 + along * 0.060,
         flange_at + axis_dir * 0.115 - across * 0.040 + along * 0.018,
         0.020, mats["torch"])

    return wire_tip


def build_camera_stand(mats):
    """The depth camera, on a stand, where cell.geometry actually puts it.

    It does not ride on the arm. The cell sees the joint from one fixed pose and
    changes what it sees by turning the part, not by moving the camera, so a
    camera bolted to the wrist would be a different machine from the one that was
    measured.

    That choice is not only about the picture. A camera on the wrist has to be
    carried: it comes off the payload, it adds inertia where the arm is trying to
    be precise, and it lives next to the arc in the heat and the spatter. Its
    world pose then has to be worked out from the joint encoders through a
    hand-eye transform, so every bit of joint error and backlash lands in the
    measurement, and the calibration has to be proved again after any service or
    crash. A camera on a stand is calibrated once into the robot base frame and
    stays there.

    What it costs instead is reach: one fixed pose cannot follow the seam along
    its length, and it can be occluded. That is exactly why the cell turns the
    part and why a model chooses which view to trust.
    """
    eye = Vector(LOOKAHEAD_EYE)
    aim = Vector((JOINT_X, JOINT_Y, LOOKAHEAD_AIM_Z))
    look = (aim - eye).normalized()

    # The post stands to one side and a boom reaches over, rather than standing
    # directly behind the camera. Behind is the worst place for it: that is the
    # camera's own sight line extended, so the post ends up in every view taken
    # from the same side, and a column in the middle of the approach is somewhere
    # a person has to walk anyway.
    #
    # Which side is decided by where the robot is, not by taste: the post goes on
    # the side the arm does not work from.
    lateral = Vector((-look.y, look.x, 0.0)).normalized()
    to_robot = Vector((ROBOT_BASE[0] - eye.x, ROBOT_BASE[1] - eye.y, 0.0))
    if lateral.dot(to_robot) > 0:
        lateral = -lateral
    # Close beside the camera, not flung out on a long cantilever. The camera
    # looks forward and the post is beside and below it, so it only has to clear
    # the housing, not the line of sight, and a short stub is stiffer than a long
    # arm. An earlier version put the post 900 mm out, which is a lot of lever for
    # something that has to hold a calibrated pose steady next to an arc.
    foot = Vector((eye.x + lateral.x * 0.30, eye.y + lateral.y * 0.30, 0.0))
    # The housing hangs behind the eye point; the mast and boom reach to it.
    body = eye - look * 0.105

    slab("camera_foot", foot.x - 0.17, foot.x + 0.17, foot.y - 0.17, foot.y + 0.17,
         0.0, 0.035, mats["robot_dark"])
    # A column to just above the housing and a short stub across to it. Running
    # the post higher and arching over looks like a crane and behaves like one:
    # every extra bit of overhang is more lever on a pose that has to stay put.
    head_top = body.z + 0.055
    tube("camera_post", (foot.x, foot.y, 0.035), (foot.x, foot.y, head_top),
         0.042, mats["frame"])
    tube("camera_stub", (foot.x, foot.y, head_top), (body.x, body.y, head_top),
         0.030, mats["frame"])
    tube("camera_drop", (body.x, body.y, head_top), body + Vector((0, 0, 0.030)),
         0.024, mats["robot_dark"])

    # Everything sits BEHIND the eye point, because the eye point is the front of
    # the lens: it is where light enters, not the middle of the casing. Straddling
    # it with the housing puts the camera's own body in front of its own window,
    # which is invisible in a render and shows up the moment anything traces a ray
    # from it.
    beam("camera_head", body - look * 0.055, body + look * 0.055, 0.105, 0.085,
         mats["sensor"])
    lens = eye - look * 0.026
    beam("camera_lens", lens - look * 0.024, lens + look * 0.024, 0.052, 0.040,
         mats["lens"])
    return eye
    return tip


def build_bead(mats, tip):
    """Weld metal already laid, from the bottom of the seam up to the arc.

    It follows the corner as the corner actually runs, not as the drawing draws
    it, because that is the claim the whole cell is making.
    """
    out = Vector((geo.JOINT_LOCAL[0], geo.JOINT_LOCAL[1], 0.0)).normalized()
    steps = 26
    top = tip.z - 0.012
    previous = None
    for i in range(steps + 1):
        z = SEAM_BOTTOM + 0.012 + (top - SEAM_BOTTOM - 0.012) * i / steps
        point = Vector((JOINT_X, JOINT_Y, z)) + out * 0.0016
        if previous is not None:
            hot = i > steps - 4
            tube("bead", previous, point, 0.0042,
                 mats["bead_hot"] if hot else mats["bead"], faces=10)
        previous = point


def build_room(mats):
    slab("floor", -3.0, 9.0, -6.0, 6.0, -0.02, 0.0, mats["floor"])
    for x in (-2.2, 8.2):
        slab("guard_post", x - 0.04, x + 0.04, -3.4, -3.32, 0.0, 1.8, mats["guard"])
    slab("guard_rail", -2.24, 8.24, -3.40, -3.32, 1.00, 1.08, mats["guard"])
    slab("guard_rail_low", -2.24, 8.24, -3.40, -3.32, 0.45, 0.53, mats["guard"])


def build_lighting():
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.055, 0.060, 0.070, 1.0)
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

    def lamp(name, location, energy, size, rotation=(0, 0, 0)):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.size = size
        data.shape = "SQUARE"
        obj = bpy.data.objects.new(name, data)
        obj.location = location
        obj.rotation_euler = Euler(rotation, "XYZ")
        bpy.context.scene.collection.objects.link(obj)
        return obj

    # A shop floor is lit from a long way up by a few big fittings, so the light
    # is soft and there is not much of it reaching a horizontal surface. Driving it
    # harder than this washes the metal out to paper white and the picture stops
    # telling you anything about the part.
    lamp("CELL_key", (geo.FIXTURE_X + 0.6, -2.4, 3.6), 260, 2.6,
         (math.radians(38), 0, math.radians(24)))
    lamp("CELL_fill", (geo.FIXTURE_X - 2.4, 2.6, 3.0), 110, 3.4,
         (math.radians(-40), 0, math.radians(-28)))
    lamp("CELL_rim", (geo.FIXTURE_X + 2.8, 1.9, 2.3), 90, 1.6,
         (math.radians(-56), 0, math.radians(118)))

    # The arc itself, a small hot source down at the seam.
    arc = bpy.data.lights.new("CELL_arc", "POINT")
    arc.energy = 9
    arc.shadow_soft_size = 0.01
    arc.color = (0.72, 0.84, 1.0)
    obj = bpy.data.objects.new("CELL_arc", arc)
    obj.location = (JOINT_X + 0.02, JOINT_Y - 0.02, (SEAM_BOTTOM + SEAM_TOP) / 2 - 0.02)
    bpy.context.scene.collection.objects.link(obj)


def aim_camera(eye, target, lens=45.0):
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type = "PERSP"
    cam.data.lens = lens
    cam.location = Vector(eye)
    direction = Vector(target) - Vector(eye)
    cam.rotation_mode = "QUATERNION"
    cam.rotation_quaternion = direction.to_track_quat("-Z", "Y")
    cam.hide_render = False
    bpy.context.scene.camera = cam
    return cam


def wipe():
    for obj in list(bpy.data.objects):
        if obj.type in {"MESH", "LIGHT", "EMPTY"}:
            bpy.data.objects.remove(obj, do_unlink=True)


def setup_render(width=1600, height=900, samples=110):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs is not None:
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.preferences.compute_device_type = kind
                prefs.preferences.get_devices()
                for device in prefs.preferences.devices:
                    device.use = device.type != "CPU"
                scene.cycles.device = "GPU"
                break
            except Exception:
                continue
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.4


def build():
    wipe()
    mats = build_materials()
    tray_forming = forming_lib.plan_tray(random.Random(SEED), scale=FORMING_SCALE)

    build_room(mats)
    build_conveyor(mats)
    build_fixture(mats)
    build_part(mats, tray_forming, geo.FIXTURE_X + 0.008, -0.006)
    tip = build_robot(mats)
    build_bead(mats, tip)
    build_camera_stand(mats)
    build_lighting()
    setup_render()
    return tray_forming, tip


# Blender executes this file rather than importing it, and the name it runs under
# is not __main__, so the usual guard would never fire. Building unconditionally is
# the intent: the file exists to be run.
tray_forming, tip = build()
result = {
    "seam_from": [round(JOINT_X, 4), round(JOINT_Y, 4), round(SEAM_BOTTOM, 4)],
    "seam_to": [round(JOINT_X, 4), round(JOINT_Y, 4), round(SEAM_TOP, 4)],
    "torch_tip": [round(v, 4) for v in tip],
    "forming": tray_forming.describe(),
    "objects": len([o for o in bpy.data.objects if o.type == "MESH"]),
}
