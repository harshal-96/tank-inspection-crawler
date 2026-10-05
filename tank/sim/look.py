"""Turn the measured scene into one worth filming, without touching the dataset.

    python scripts/blender_send.py tank/sim/blender_tank.py   # build it
    python scripts/blender_send.py tank/sim/look.py           # make it look real

Run after the build. Nothing here changes a single vertex, a defect, a camera
pose or a ground truth number: it replaces shaders, lights and render settings
only. The 24 scored frames in tank/data/survey stay exactly as they were
measured, because a dataset whose appearance changes every time somebody
improves the lighting is not a dataset.

What actually makes a render stop looking like a render, in the order it
matters, which is not the order people usually reach for.

1. The renderer. The scene was on EEVEE, which rasterises and fakes its
   bounces. Inside a steel drum almost every photon that reaches the camera
   has bounced off something first, so faking the bounces is faking the
   picture. Cycles path traces it on the GPU through OptiX.
2. The air. A tank is not a vacuum. Give it a faint haze and the work lights
   become visible shafts, distance reads as distance, and the far wall falls
   away properly. One scatter node does more for realism here than a million
   polygons would.
3. The surfaces. Rust is not orange, it is twenty browns with black pits and a
   crust you can feel. Steel is not grey, it is a mirror that has been shot
   blasted. Both need roughness that varies across the surface, because a
   constant roughness is the single clearest tell of a synthetic image.
4. The lens. Depth of field at a hand's distance is severe, and a picture
   where everything is equally sharp was not taken by a camera.

Polygon count is nowhere on that list. The geometry was already dense enough.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hero stills want the samples; an animation pass can drop them a long way and
# let the denoiser carry it, because noise that moves is worse than noise that
# sits still.
HERO_SAMPLES = 1024
HERO_RES = (1920, 1080)
NOISE_FLOOR = 0.008        # adaptive sampling stops early on clean pixels
HAZE = 0.012               # scatter density per metre, inside the shell
# At 126 mm an f/4 lens has a few millimetres of depth of field and the
# whole frame turns to mush. A real inspection camera is stopped well down
# precisely so the operator can see the surface.
F_STOP = 22.0


def _nodes(mat):
    mat.use_nodes = True
    return mat.node_tree, mat.node_tree.nodes, mat.node_tree.links


def _clear(mat):
    tree, nodes, _ = _nodes(mat)
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    out.location = (600, 0)
    bsdf.location = (300, 0)
    tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return tree, nodes, tree.links, bsdf


def _coords(nodes, links, target):
    """Object space, so a noise scale is a real number of metres."""
    tex = nodes.new("ShaderNodeTexCoord")
    tex.location = (-1200, 0)
    links.new(tex.outputs["Object"], target.inputs["Vector"])
    return tex


def _noise(nodes, links, coords, scale, detail=8.0, roughness=0.55, y=0):
    n = nodes.new("ShaderNodeTexNoise")
    n.location = (-950, y)
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = roughness
    links.new(coords.outputs["Object"], n.inputs["Vector"])
    return n


def _ramp(nodes, links, source, stops, y=0):
    r = nodes.new("ShaderNodeValToRGB")
    r.location = (-700, y)
    links.new(source, r.inputs["Fac"])
    ramp = r.color_ramp
    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[-1])
    first, *rest = stops
    ramp.elements[0].position, ramp.elements[0].color = first[0], (*first[1], 1.0)
    for pos, col in rest:
        e = ramp.elements.new(pos)
        e.color = (*col, 1.0)
    return r


def _bump_chain(nodes, links, coords, bsdf, scales):
    """Several scales of relief, each feeding the next.

    One scale of bump is always wrong somewhere: coarse enough to read at a
    metre and it is mush at a hand's distance, fine enough to read close and it
    vanishes across the room. They stack instead.
    """
    previous = None
    for index, (scale, strength, detail) in enumerate(scales):
        n = _noise(nodes, links, coords, scale, detail=detail, y=-400 - index * 240)
        b = nodes.new("ShaderNodeBump")
        b.location = (-420, -400 - index * 240)
        b.inputs["Strength"].default_value = strength
        b.inputs["Distance"].default_value = 0.4 / max(scale, 1.0)
        links.new(n.outputs["Fac"], b.inputs["Height"])
        if previous is not None:
            links.new(previous.outputs["Normal"], b.inputs["Normal"])
        previous = b
    if previous is not None:
        links.new(previous.outputs["Normal"], bsdf.inputs["Normal"])


def _set(bsdf, name, value):
    socket = bsdf.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def steel(mat, base, tint, rough_lo, rough_hi, coarse=26.0, fine=900.0):
    """Shot blasted plate. A mirror that has been sandblasted, not grey paint."""
    tree, nodes, links, bsdf = _clear(mat)
    coords = _coords(nodes, links, nodes.new("ShaderNodeTexNoise"))
    nodes.remove([n for n in nodes if n.type == "TEX_NOISE"][0])

    mottle = _noise(nodes, links, coords, coarse * 0.35, detail=6.0, y=200)
    colour = _ramp(nodes, links, mottle.outputs["Fac"],
                   [(0.30, base), (0.70, tint)], y=200)
    links.new(colour.outputs["Color"], bsdf.inputs["Base Color"])

    # Roughness that varies is the whole game. A constant one is the clearest
    # tell there is that nobody stood in front of this.
    grain = _noise(nodes, links, coords, coarse * 1.9, detail=9.0, y=-60)
    rough = _ramp(nodes, links, grain.outputs["Fac"],
                  [(0.25, (rough_lo,) * 3), (0.78, (rough_hi,) * 3)], y=-60)
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    _set(bsdf, "Metallic", 1.0)
    _bump_chain(nodes, links, coords,
                bsdf, ((coarse, 0.22, 6.0), (fine * 0.25, 0.14, 8.0),
                       (fine, 0.09, 4.0)))
    return mat


def rust(mat):
    """Oxide, not paint. Twenty browns, a crust, and almost no metal left."""
    tree, nodes, links, bsdf = _clear(mat)
    probe = nodes.new("ShaderNodeTexNoise")
    coords = _coords(nodes, links, probe)
    nodes.remove(probe)

    # Two scales of colour variation. One alone leaves a flat field: the
    # coarse one gives the patch its shape, the fine one is the grit you see
    # at a hand's distance, and without the second it is orange paint.
    patchy = _noise(nodes, links, coords, 22.0, detail=12.0, roughness=0.72, y=320)
    grit = _noise(nodes, links, coords, 420.0, detail=10.0, roughness=0.62, y=440)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "FLOAT"
    mix.location = (-820, 380)
    mix.inputs[0].default_value = 0.42
    links.new(patchy.outputs["Fac"], mix.inputs[2])
    links.new(grit.outputs["Fac"], mix.inputs[3])
    colour = _ramp(nodes, links, mix.outputs[0], [
        (0.04, (0.030, 0.012, 0.006)),      # the near black of a deep pit
        (0.26, (0.105, 0.038, 0.018)),      # old dark scale
        (0.48, (0.225, 0.082, 0.030)),      # the brown most of it actually is
        (0.68, (0.330, 0.135, 0.048)),      # the orange everybody draws
        (0.88, (0.440, 0.245, 0.120)),      # powdery, where it is drying out
    ], y=200)
    # Part of the ramp steps rather than blends. Oxide comes off in flakes with
    # edges, and a pure gradient reads as an airbrushed wash.
    colour.color_ramp.interpolation = "B_SPLINE"
    links.new(colour.outputs["Color"], bsdf.inputs["Base Color"])

    grain = _noise(nodes, links, coords, 140.0, detail=10.0, y=-60)
    rough = _ramp(nodes, links, grain.outputs["Fac"],
                  [(0.20, (0.78,) * 3), (0.85, (1.0,) * 3)], y=-60)
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    # Rust is an oxide. Leaving it metallic is why it read as orange plastic.
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Specular IOR Level", 0.22)
    _bump_chain(nodes, links, coords,
                bsdf, ((55.0, 0.55, 8.0), (260.0, 0.38, 10.0), (900.0, 0.18, 6.0)))
    return mat


def bead(mat):
    """Weld metal: brighter than the plate, rougher along the ripple valleys."""
    tree, nodes, links, bsdf = _clear(mat)
    probe = nodes.new("ShaderNodeTexNoise")
    coords = _coords(nodes, links, probe)
    nodes.remove(probe)

    mottle = _noise(nodes, links, coords, 70.0, detail=8.0, y=200)
    colour = _ramp(nodes, links, mottle.outputs["Fac"], [
        (0.25, (0.148, 0.140, 0.128)),
        (0.60, (0.235, 0.222, 0.198)),
        (0.85, (0.300, 0.282, 0.250)),
    ], y=200)
    links.new(colour.outputs["Color"], bsdf.inputs["Base Color"])

    grain = _noise(nodes, links, coords, 320.0, detail=10.0, y=-60)
    rough = _ramp(nodes, links, grain.outputs["Fac"],
                  [(0.22, (0.26,) * 3), (0.80, (0.58,) * 3)], y=-60)
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    _set(bsdf, "Metallic", 0.88)
    _bump_chain(nodes, links, coords, bsdf, ((220.0, 0.20, 8.0), (1100.0, 0.12, 5.0)))
    return mat


def dark(mat):
    """The floor of a pit or a groove. Nearly black, and never shiny."""
    tree, nodes, links, bsdf = _clear(mat)
    # Not absolute black. Nothing in a lit tank is, and a pure black floor
    # renders as a flat sticker rather than as a hole with a bottom.
    _set(bsdf, "Base Color", (0.040, 0.034, 0.030, 1.0))
    _set(bsdf, "Roughness", 0.88)
    _set(bsdf, "Metallic", 0.0)
    return mat


def painted(mat, colour, wear=(0.26, 0.24, 0.22)):
    """Industrial paint over steel, with the paint losing.

    A machine that has been inside a tank is not the colour it left the factory.
    Flat colour on the hero object is what makes a render look like a product
    shot of a toy, so the coat is scuffed back to primer on the edges and the
    gloss is uneven because nothing stays evenly glossy underground.
    """
    tree, nodes, links, bsdf = _clear(mat)
    probe = nodes.new("ShaderNodeTexNoise")
    coords = _coords(nodes, links, probe)
    nodes.remove(probe)

    # Wear concentrated where a noise happens to be high, standing in for the
    # corners and edges that actually take the knocks.
    scuff = _noise(nodes, links, coords, 38.0, detail=10.0, roughness=0.70, y=200)
    colour_ramp = _ramp(nodes, links, scuff.outputs["Fac"], [
        (0.36, colour),
        (0.62, tuple(c * 0.72 for c in colour)),
        (0.80, wear),
    ], y=200)
    links.new(colour_ramp.outputs["Color"], bsdf.inputs["Base Color"])

    gloss = _noise(nodes, links, coords, 95.0, detail=8.0, y=-60)
    rough = _ramp(nodes, links, gloss.outputs["Fac"],
                  [(0.25, (0.24,) * 3), (0.80, (0.52,) * 3)], y=-60)
    links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])

    _set(bsdf, "Metallic", 0.15)
    _set(bsdf, "Coat Weight", 0.35)
    _set(bsdf, "Coat Roughness", 0.22)
    _bump_chain(nodes, links, coords, bsdf, ((160.0, 0.14, 8.0), (900.0, 0.09, 5.0)))
    return mat


def glass(mat):
    """A lens behind a port: dark, smooth, and it has to catch a highlight."""
    tree, nodes, links, bsdf = _clear(mat)
    _set(bsdf, "Base Color", (0.015, 0.022, 0.032, 1.0))
    _set(bsdf, "Roughness", 0.06)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Coat Weight", 1.0)
    _set(bsdf, "Coat Roughness", 0.02)
    _set(bsdf, "Specular IOR Level", 0.8)
    return mat


def rubber(mat):
    """A tyre that has been run on steel: dull, dusty, never pure black."""
    tree, nodes, links, bsdf = _clear(mat)
    probe = nodes.new("ShaderNodeTexNoise")
    coords = _coords(nodes, links, probe)
    nodes.remove(probe)
    grain = _noise(nodes, links, coords, 260.0, detail=9.0, y=200)
    colour = _ramp(nodes, links, grain.outputs["Fac"],
                   [(0.30, (0.016, 0.016, 0.018)), (0.82, (0.046, 0.044, 0.044))],
                   y=200)
    links.new(colour.outputs["Color"], bsdf.inputs["Base Color"])
    _set(bsdf, "Roughness", 0.86)
    _set(bsdf, "Metallic", 0.0)
    _bump_chain(nodes, links, coords, bsdf, ((420.0, 0.30, 8.0), (1800.0, 0.16, 5.0)))
    return mat


def atmosphere(density=HAZE):
    """Put air in the tank.

    Light inside a sealed steel drum is carried by what is floating in it. With
    a vacuum the work lights are two bright discs and everything between them
    is black; with a little haze they throw shafts, the far wall recedes, and
    the shot acquires the one thing a render usually lacks, which is distance.
    """
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    for n in list(nodes):
        nodes.remove(n)
    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (300, 0)

    # Almost no ambient. A closed tank has no sky in it, and faking one is how
    # an interior ends up looking like a studio.
    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.012, 0.014, 0.018, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.location = (0, -200)
    scatter.inputs["Color"].default_value = (0.72, 0.74, 0.80, 1.0)
    scatter.inputs["Density"].default_value = density
    scatter.inputs["Anisotropy"].default_value = 0.35
    links.new(scatter.outputs["Volume"], out.inputs["Volume"])
    return world


def lights(warm=(1.0, 0.93, 0.82)):
    """Keep the rig, fix the quality of it.

    Spots with a hard edge read as stage lighting. Real floods on a tank floor
    are big, soft and warm, and they bounce off the shell far more than they
    light anything directly.
    """
    touched = 0
    for obj in bpy.data.objects:
        if obj.type != "LIGHT":
            continue
        lamp = obj.data
        lamp.color = warm
        lamp.use_shadow = True
        if lamp.type == "SPOT":
            lamp.spot_blend = max(lamp.spot_blend, 0.92)
            lamp.shadow_soft_size = max(lamp.shadow_soft_size, 0.35)
        elif lamp.type == "POINT":
            lamp.shadow_soft_size = max(lamp.shadow_soft_size, 0.04)
        touched += 1
    return touched


def lens(camera_name="Crawler_Cam", f_stop=F_STOP):
    """Depth of field, because a hand's distance has very little of it."""
    cam = bpy.data.objects.get(camera_name)
    if cam is None or cam.type != "CAMERA":
        return None
    data = cam.data
    data.dof.use_dof = True
    data.dof.aperture_fstop = f_stop
    # Focus on the wall the lens is pointed at rather than a guessed distance.
    data.dof.focus_distance = max(0.02, data.dof.focus_distance or 0.126)
    return cam


def render_settings(samples=HERO_SAMPLES, res=HERO_RES):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
    except Exception:
        prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type != "CPU"
    scene.cycles.device = "GPU"

    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = NOISE_FLOOR
    scene.cycles.use_denoising = True
    with contextlib.suppress(Exception):
        scene.cycles.denoiser = "OPTIX"
    # Bounces: a steel drum is a box of mirrors, so starving the glossy path is
    # what makes an interior look flat and dead.
    scene.cycles.max_bounces = 12
    scene.cycles.glossy_bounces = 8
    scene.cycles.transmission_bounces = 8
    scene.cycles.volume_bounces = 2
    scene.cycles.caustics_reflective = False

    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_motion_blur = True
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium Contrast"
    except Exception:
        pass
    return scene


MATERIALS = {
    "tank_plate": lambda m: steel(m, (0.052, 0.056, 0.064), (0.098, 0.104, 0.118),
                                  0.28, 0.62),
    "weld_bead": bead,
    "corrosion": rust,
    "defect_dark": dark,
    "heat_affected_zone": lambda m: steel(m, (0.030, 0.026, 0.021),
                                          (0.062, 0.052, 0.040), 0.42, 0.74,
                                          coarse=60.0, fine=1200.0),
    "crawler_frame": lambda m: steel(m, (0.022, 0.024, 0.028),
                                     (0.048, 0.050, 0.056), 0.30, 0.52,
                                     coarse=120.0, fine=1600.0),
    "crawler_hub": lambda m: steel(m, (0.30, 0.31, 0.33), (0.52, 0.53, 0.56),
                                   0.16, 0.34, coarse=200.0, fine=2000.0),
    # The hero object. Left as flat tan it was the one thing in every frame
    # that looked moulded rather than built.
    "crawler_body": lambda m: painted(m, (0.62, 0.185, 0.018)),
    "crawler_tyre": rubber,
    "lens_glass": glass,
}


def main() -> dict:
    done = []
    for name, build in MATERIALS.items():
        mat = bpy.data.materials.get(name)
        if mat is not None:
            build(mat)
            done.append(name)

    world = atmosphere()
    lit = lights()
    cam = lens()
    scene = render_settings()
    return {
        "materials_rebuilt": done,
        "lights_softened": lit,
        "depth_of_field_on": bool(cam),
        "haze_density": HAZE,
        "engine": scene.render.engine,
        "device": scene.cycles.device,
        "samples": scene.cycles.samples,
        "resolution": list(HERO_RES),
        "world": world.name,
        "note": "dataset frames in tank/data/survey are untouched by this",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=1))
