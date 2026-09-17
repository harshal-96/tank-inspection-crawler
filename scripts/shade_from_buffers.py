"""Shade a clean image from the depth and normal buffers.

Isaac's colour pass draws large black bands on faces that are edge-on to the dome
light, and in this headless configuration no analytic light contributes anything,
so nothing in the scene can fill them in. The geometry buffers do not have that
problem: depth and normals are correct everywhere, which is why measurement works
off these same captures.

So the picture is built from those instead. Lambertian shading from a chosen light
direction, a little ambient, and a depth-based fade for context. No renderer
involved in the lighting at all.

    python scripts/shade_from_buffers.py --manifest data/captures/manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Directions FROM the surface TO each light. The camera views the joint from +x
# and -y, so the key has to come from that side or the two faces it sees get
# nothing but ambient. The fill comes over the opposite shoulder so nothing that
# faces away from the key falls out completely.
KEY_DIRECTION = np.array([0.55, -0.68, 0.48])
FILL_DIRECTION = np.array([-0.35, -0.30, 0.88])
KEY_STRENGTH = 0.85
FILL_STRENGTH = 0.50
AMBIENT = 0.38

BASE_COLOUR = np.array([0.80, 0.815, 0.835])
BACKGROUND = np.array([0.93, 0.935, 0.94])


def decode_normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Normal buffer to unit vectors, plus a mask of where a normal was written.

    Stored as 0..255 over the range -1 to 1. Where nothing was rendered the buffer
    holds 127, which decodes to -0.004 rather than to zero. Normalising that tiny
    vector turns empty background into a confident unit normal pointing at
    (-1,-1,-1), which then shades as though it were a surface. The length has to be
    checked before normalising, not after.
    """
    raw = np.asarray(Image.open(path))[..., :3].astype(np.float64)
    vectors = raw / 255.0 * 2.0 - 1.0
    lengths = np.linalg.norm(vectors, axis=2, keepdims=True)
    written = (lengths[..., 0] > 0.2)
    unit = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0.2)
    return unit, written


def shade(normals: np.ndarray, written: np.ndarray, depth: np.ndarray) -> np.ndarray:
    surface = np.isfinite(depth) & (depth > 0) & written

    key = np.clip(normals @ (KEY_DIRECTION / np.linalg.norm(KEY_DIRECTION)), 0.0, 1.0)
    fill = np.clip(normals @ (FILL_DIRECTION / np.linalg.norm(FILL_DIRECTION)), 0.0, 1.0)
    light = AMBIENT + KEY_STRENGTH * key + FILL_STRENGTH * fill

    image = BASE_COLOUR[None, None, :] * light[..., None]

    # Fade the far field so the part reads against its surroundings.
    if surface.any():
        near, far = np.percentile(depth[surface], [2, 98])
        span = max(far - near, 1e-6)
        haze = np.clip((depth - near) / span, 0.0, 1.0)[..., None]
        image = image * (1.0 - 0.45 * haze) + BACKGROUND[None, None, :] * (0.45 * haze)

    image = np.where(surface[..., None], image, BACKGROUND[None, None, :])
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def edges(depth: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """Real depth discontinuities only.

    A fixed millimetre threshold does not work: a flat surface seen obliquely
    changes depth by tens of millimetres per pixel, so a low threshold paints the
    whole surface as an edge. What separates a true silhouette from a steep
    surface is how far the step stands out from the steps around it.
    """
    filled = np.where(np.isfinite(depth), depth, np.nan)
    gx = np.abs(np.gradient(filled, axis=1))
    gy = np.abs(np.gradient(filled, axis=0))
    step = np.nan_to_num(np.maximum(gx, gy))
    finite = step[np.isfinite(step) & (step > 0)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=bool)
    median = float(np.median(finite))
    spread = float(np.median(np.abs(finite - median))) or 1e-6
    return step > median + sigma * spread


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/captures/manifest.json")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--view", type=float, default=0.0)
    parser.add_argument("--out", default="data/shaded.png")
    parser.add_argument(
        "--edges",
        action="store_true",
        help="Overlay depth discontinuities. Off by default: on a surface seen "
        "obliquely it still marks far more than the real silhouettes.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    row = next(r for r in manifest["rows"] if r["trial"] == args.part)
    view = min(row["views"], key=lambda v: abs(v["orbit_deg"] - args.view))

    depth = np.load(root / view["depth"])
    normals, written = decode_normals(root / view["normals"])
    image = shade(normals, written, depth)

    if args.edges:
        image[edges(depth)] = (48, 52, 58)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.out)

    black = (image.astype(int).sum(axis=2) < 90).mean()
    print(f"part {args.part} view {view['orbit_deg']:+.1f} deg  ->  {args.out}")
    print(f"black pixels: {100 * black:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
