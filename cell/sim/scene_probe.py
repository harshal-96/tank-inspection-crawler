"""Build the scene one piece at a time and measure where the black appears.

A diagnostic, not part of the cell. The full scene renders hard black bands across
the table, the panels and the stop, and every property-level test came back
identical: lighting, sample count, normals, materials, semantics, geometry type.
That pattern says the cause is not a property of one prim but something about the
scene, so the scene gets bisected instead.

    E:\\Issac-sim\\python.bat cell\\sim\\scene_probe.py --stage 0

Stages are cumulative:
    0  cabinet and camera only
    1  + ground
    2  + fixture table
    3  + end stop
    4  + guide rails
    5  + weld bead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", type=int, default=0)
parser.add_argument("--out", default="data/probe")
parser.add_argument("--dome", type=float, default=3000.0)
parser.add_argument("--solid", action="store_true",
                    help="One box instead of six overlapping panels")
parser.add_argument("--rendermode", default=None,
                    help="RaytracedLighting or PathTracing")
ARGS = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402

config = {"headless": True}
if ARGS.rendermode:
    config["renderer"] = ARGS.rendermode
app = SimulationApp(config)
if ARGS.rendermode:
    import carb
    carb.settings.get_settings().set("/rtx/rendermode", ARGS.rendermode)

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux  # noqa: E402

from cell import geometry as geo  # noqa: E402

OUT = (PROJECT_ROOT / ARGS.out / f"stage{ARGS.stage}").resolve()
OUT.mkdir(parents=True, exist_ok=True)

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def box(path, size, translate, color=(0.62, 0.64, 0.68)):
    prim = UsdGeom.Cube.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*translate))
    prim.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return prim


def build_cabinet(root: str, at_x: float):
    xform = UsdGeom.Xform.Define(stage, root)
    xform.AddTranslateOp().Set(Gf.Vec3d(at_x, 0.0, geo.TABLE_TOP))
    w, h, d, t = geo.PART_W, geo.PART_H, geo.PART_D, geo.PART_T

    if ARGS.solid:
        # From outside, a closed cabinet and a solid block are the same object.
        # The weld joint is the outer corner edge either way, and a single prim
        # has no interior to show through and no panels to intersect.
        box(f"{root}/body", (w, h, d), (0, 0, d / 2))
        return
    for name, size, offset in (
        ("bottom", (w, h, t), (0, 0, t / 2)),
        ("top", (w, h, t), (0, 0, d - t / 2)),
        ("front", (w, t, d), (0, -(h / 2 - t / 2), d / 2)),
        ("back", (w, t, d), (0, (h / 2 - t / 2), d / 2)),
        ("left", (t, h, d), (-(w / 2 - t / 2), 0, d / 2)),
        ("right", (t, h, d), ((w / 2 - t / 2), 0, d / 2)),
    ):
        box(f"{root}/{name}", size, offset)


def main() -> int:
    build_cabinet("/World/Cabinet", geo.FIXTURE_X)

    if ARGS.stage >= 1:
        box("/World/Ground", (40, 40, 0.02), (0, 0, -0.01), (0.22, 0.22, 0.24))
    if ARGS.stage >= 2:
        box("/World/Table", (1.6, 1.7, 0.08), (geo.FIXTURE_X, 0, geo.TABLE_TOP - 0.04),
            (0.55, 0.57, 0.60))
    if ARGS.stage >= 3:
        box("/World/Stop", (0.05, 1.3, 0.12),
            (geo.FIXTURE_X + 0.485, 0, geo.TABLE_TOP + 0.06), (0.50, 0.20, 0.10))
    if ARGS.stage >= 4:
        for index, side in ((0, -1), (1, 1)):
            rail = box(f"/World/Guide_{index}", (7.0, 0.04, 0.10),
                       (0.8, side * 0.66, geo.TABLE_TOP + 0.05))
            UsdGeom.Imageable(rail).CreatePurposeAttr().Set("guide")
    if ARGS.stage >= 5:
        bw, bd, length = geo.BEAD_SIZE
        box("/World/Cabinet/bead", (bw, bd, length),
            (geo.JOINT_LOCAL[0] + geo.BEAD_PROUD,
             geo.JOINT_LOCAL[1] - geo.BEAD_PROUD,
             geo.BEAD_Z), (0.78, 0.68, 0.48))

    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr().Set(ARGS.dome)

    camera = UsdGeom.Camera.Define(stage, "/World/Cam")
    camera.CreateFocalLengthAttr().Set(24.0)
    camera.CreateHorizontalApertureAttr().Set(20.955)
    camera.CreateVerticalApertureAttr().Set(20.955)
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*geo.CAMERA_EYE), Gf.Vec3d(*geo.CAMERA_AIM), Gf.Vec3d(0, 0, 1)
    )
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())

    product = rep.create.render_product(str(camera.GetPath()), (640, 640))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(OUT).replace("\\", "/"), rgb=True)
    writer.attach([product])

    rep.orchestrator.step(rt_subframes=16)
    rep.orchestrator.step(rt_subframes=16)
    rep.orchestrator.wait_until_complete()

    frame = OUT / "rgb_0001.png"
    if frame.exists():
        from PIL import Image

        pixels = np.asarray(Image.open(frame).convert("RGB")).astype(int)
        black = (pixels.sum(axis=2) < 90).mean()
        print(f"STAGE {ARGS.stage}: black {100 * black:.2f}%   -> {frame}", flush=True)
    else:
        print(f"STAGE {ARGS.stage}: no frame written", flush=True)
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        app.close()
    raise SystemExit(code)
