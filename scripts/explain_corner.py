"""Colour every surface at the weld corner by what it actually is.

The shaded picture is honest but it is all one shade of grey, so a line at the
corner could be the joint, the edge of a panel, or something drawn on top. Here
each surface is coloured by which way it faces, taken from the normals buffer, and
each overlay gets its own colour and a swatch in the key. Nothing in the picture
is left for the eye to guess at.

    python scripts/explain_corner.py --manifest data/captures_seam/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell import geometry as geo  # noqa: E402
from cell.capture import load_depth  # noqa: E402
from cell.depth_joint import Intrinsics, backproject, project, transform_points  # noqa: E402
from cell.seam_track import find_and_track  # noqa: E402
from scripts.shade_from_buffers import decode_normals  # noqa: E402

BACKDROP = (250, 251, 252)
INK = (26, 30, 36)
MUTED = (122, 132, 144)

LONG_SIDE = (66, 148, 166)      # the long side of the tray, facing the camera
END_WALL = (150, 110, 186)      # the end wall it meets at this corner
FLAT = (222, 206, 172)          # anything horizontal: rim, base, table
NOTHING = (238, 241, 244)       # no metal here

TAUGHT = (130, 137, 148)
TRUTH = (214, 58, 58)
MEASURED = (32, 122, 210)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/captures_seam/manifest.json")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--zoom", type=int, default=2)
    parser.add_argument("--out", default="data/corner_explained.png")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    camera = manifest["camera"]
    row = next(r for r in manifest["rows"] if r["trial"] == args.part)
    view = row["views"][0]

    intrinsics = Intrinsics.from_camera(
        camera["width"], camera["height"], camera["focal_length_mm"],
        camera["horizontal_aperture_mm"], camera["vertical_aperture_mm"],
    )
    to_world = np.array(camera["to_world"])

    depth = load_depth(root / view["depth"])
    normals, _ = decode_normals(root / view["normals"])
    height, width = depth.shape

    # Colour by which way each surface faces. That is the whole point: the two
    # faces that make this joint are perpendicular, so they separate cleanly, and
    # anything horizontal is fixture or rim rather than joint.
    painted = np.zeros((height, width, 3), dtype=np.uint8)
    painted[:] = NOTHING
    metal = np.isfinite(depth)
    facing_x = metal & (np.abs(normals[..., 0]) > 0.7)
    facing_y = metal & (np.abs(normals[..., 1]) > 0.7)
    facing_z = metal & (np.abs(normals[..., 2]) > 0.7)
    painted[facing_z] = FLAT
    painted[facing_y] = END_WALL
    painted[facing_x] = LONG_SIDE

    # Shade each one a little by range, so the picture still reads as a solid
    # object rather than as flat blocks of colour.
    near, far = np.nanpercentile(depth[metal], [2, 98])
    lit = np.clip((far - depth) / max(far - near, 1e-6), 0.0, 1.0)
    lit = (0.72 + 0.38 * lit)[..., None]
    painted[metal] = np.clip(painted[metal] * lit[metal], 0, 255).astype(np.uint8)

    cloud = transform_points(backproject(depth, intrinsics), to_world)
    floor = geo.TABLE_TOP + geo.PART_T
    nominal = geo.nominal_joint()
    seam_start = np.array([nominal[0], nominal[1], floor + 0.04])
    seam_end = np.array([nominal[0], nominal[1], floor + geo.PART_D - 0.02])
    track = find_and_track(cloud, seam_start, seam_end, stations=20)
    truth = np.array(row["true_seam"])

    # Crop to the corner, because that is the only part of the frame in question.
    centre, _ = project(np.array([truth[len(truth) // 2]]), intrinsics, to_world)
    cx = int(centre[0][0])
    half = 110
    left = max(0, min(cx - half, width - 2 * half))
    crop = (left, 60, left + 2 * half, height - 40)

    zoom = args.zoom
    panel = Image.fromarray(painted).crop(crop)
    panel = panel.resize((panel.width * zoom, panel.height * zoom), Image.NEAREST)
    draw_panel = ImageDraw.Draw(panel)

    def to_panel(points):
        uv, _ = project(np.asarray(points), intrinsics, to_world)
        return [((u - crop[0]) * zoom, (v - crop[1]) * zoom) for u, v in uv]

    draw_panel.line(to_panel([seam_start, seam_end]), fill=TAUGHT, width=3)

    keep = truth[truth[:, 2] >= seam_start[2]]
    if len(keep) > 1:
        draw_panel.line(to_panel(keep), fill=TRUTH, width=5)

    for point in to_panel([s.position for s in track.measured]):
        draw_panel.ellipse(
            [point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6],
            fill=MEASURED, outline=(255, 255, 255), width=2,
        )

    errors = []
    for station in track.measured:
        nearest = min(truth, key=lambda p: abs(p[2] - station.position[2]))
        errors.append(float(np.linalg.norm(nearest[:2] - station.position[:2]) * 1000.0))

    # -------------------------------------------------------------------- page
    pad = 28
    key_width = 470
    page = Image.new(
        "RGB", (panel.width + key_width + pad * 3, panel.height + pad * 2), BACKDROP
    )
    page.paste(panel, (pad, pad))
    draw = ImageDraw.Draw(page)
    draw.rectangle([pad, pad, pad + panel.width, pad + panel.height], outline=(214, 220, 227))

    x = pad * 2 + panel.width
    y = pad

    def heading(text):
        nonlocal y
        draw.text((x, y), text, fill=INK)
        y += 22

    def swatch(colour, title, body, shape="block"):
        nonlocal y
        if shape == "block":
            draw.rectangle([x, y + 2, x + 26, y + 14], fill=colour, outline=(190, 198, 206))
        elif shape == "line":
            draw.line([(x, y + 8), (x + 26, y + 8)], fill=colour, width=5)
        else:
            draw.ellipse(
                [x + 7, y + 1, x + 21, y + 15], fill=colour,
                outline=(255, 255, 255), width=2,
            )
        draw.text((x + 38, y), title, fill=INK)
        y += 17
        for line in body:
            draw.text((x + 38, y), line, fill=MUTED)
            y += 15
        y += 12

    heading("WHAT YOU ARE LOOKING AT")
    y += 6
    swatch(LONG_SIDE, "the long side of the tray", [
        "its outer face, standing up towards you.",
        "every pixel here faces along x.",
    ])
    swatch(END_WALL, "the end wall", [
        "the side it meets at this corner.",
        "every pixel here faces along y.",
    ])
    swatch(FLAT, "horizontal metal", [
        "the rim on top of the walls, the base",
        "inside, and the fixture table.",
    ])
    swatch(NOTHING, "no metal", [
        "background. nothing was in this pixel.",
    ])

    y += 6
    heading("DRAWN ON TOP, NOT PART OF THE TRAY")
    y += 6
    swatch(TAUGHT, "the taught path", [
        "this is the grey line you asked about.",
        "it is where a fixed program drives.",
        "it is not on the part at all: it is the",
        "robot's belief about where the part is.",
    ], shape="line")
    swatch(TRUTH, "where the seam really is", [
        "known only because this is simulated.",
    ], shape="line")
    swatch(MEASURED, "measured from depth", [
        "one reading per station up the seam.",
    ], shape="dot")

    y += 6
    heading("THE NUMBERS")
    y += 6
    lines = [
        (f"part {args.part}, {len(track.measured)} of 20 stations read", False),
        (f"the taught path misses by {row['taught_miss_mm']:.1f} mm", True),
        (f"measured lands within {np.mean(errors):.2f} mm on average, "
         f"{np.max(errors):.2f} mm at worst", True),
        (f"press left {row['forming']['right_lean_deg']:+.2f} and "
         f"{row['forming']['front_lean_deg']:+.2f} deg of lean", False),
        (f"sheet is {geo.PART_T * 1000:.0f} mm, so the notch between", False),
        (f"the two outer faces is about {geo.PART_T * 2000:.0f} mm across,", False),
        ("narrower than a pixel at this range", False),
    ]
    for line, strong in lines:
        draw.text((x, y), line, fill=INK if strong else MUTED)
        y += 17

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    page.save(args.out)
    print(f"stations {len(track.measured)}/20, mean {np.mean(errors):.2f} mm, "
          f"worst {np.max(errors):.2f} mm, taught miss {row['taught_miss_mm']:.1f} mm")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
