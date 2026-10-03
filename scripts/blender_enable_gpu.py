"""Force Cycles onto the GPU inside a headless Blender.

    blender -b shot.blend -P scripts/blender_enable_gpu.py -a

Which device Cycles uses is split across two places: the scene remembers that
it wants a GPU, but which GPU, and whether the backend is OptiX or CUDA, lives
in user preferences. A background Blender does not necessarily inherit the
preferences the interactive one was configured with, so a render launched from
a terminal can quietly fall back to the CPU and take twenty times as long while
reporting nothing wrong at all.

This states it explicitly, and prints what it ended up with so a slow render is
never a mystery.
"""

import bpy

prefs = bpy.context.preferences.addons["cycles"].preferences
chosen = None
for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
    try:
        prefs.compute_device_type = backend
        chosen = backend
        break
    except TypeError:
        continue

prefs.get_devices()
enabled = []
for device in prefs.devices:
    device.use = device.type != "CPU"
    if device.use:
        enabled.append(f"{device.type}:{device.name}")

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU" if enabled else "CPU"

print(f"[gpu] backend={chosen} device={scene.cycles.device} "
      f"samples={scene.cycles.samples} using={enabled or 'CPU only'}")
