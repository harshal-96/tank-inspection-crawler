"""Draw what the seam tracker measures, against what a taught program assumes.

Two panels. On the left, the seam seen from above: where the taught program
thinks it is, where it actually is, and where the tracker put it. On the right,
the gap along the seam, which is the press check.

    python scripts/draw_seam_track.py --bow 3 --out data/seam_track.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell.seam_track import TOUCHING_TOLERANCE_MM, track_seam  # noqa: E402

INK = (28, 32, 38)
MUTED = (140, 148, 158)
TAUGHT = (150, 155, 165)
ACTUAL = (214, 58, 58)
MEASURED = (36, 140, 214)
GOOD = (52, 160, 92)
BAD = (214, 110, 40)

SEAM_START = np.array([0.0, 0.0, 0.0])
SEAM_LENGTH = 0.40
SPACING = 0.0004


def build(bow_mm: float, gap_at, face_extent: float = 0.06, noise_mm: float = 0.0):
    rng = np.random.default_rng(0)
    points, truth = [], []
    for z in np.arange(0.0, SEAM_LENGTH, SPACING):
        t = z / SEAM_LENGTH
        bow = (bow_mm / 1000.0) * np.sin(np.pi * t)
        back = gap_at(z) / 2000.0
        truth.append((z, bow))
        for offset in np.arange(back, face_extent, SPACING):
            # Range noise acts along the line of sight, across each face, not
            # along it. A pixel either sees metal or it does not, so the edge of
            # the material stays sharp however noisy the depth is. Perturbing the
            # points in every direction instead would smear material into the gap
            # and destroy the thing being measured.
            jitter_a = rng.normal(0, noise_mm / 1000.0) if noise_mm else 0.0
            jitter_b = rng.normal(0, noise_mm / 1000.0) if noise_mm else 0.0
            points.append([bow + jitter_a, -offset, z])
            points.append([bow + offset, jitter_b, z])
    return np.array(points), np.array(truth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bow", type=float, default=3.0, help="Seam bow, mm")
    parser.add_argument("--noise", type=float, default=0.3, help="Sensor noise, mm")
    parser.add_argument("--stations", type=int, default=24)
    parser.add_argument("--out", default="data/seam_track.png")
    args = parser.parse_args()

    def gap_at(z):
        return 0.0 if z < 0.24 else 2.4 * (z - 0.24) / (SEAM_LENGTH - 0.24)

    cloud, truth = build(args.bow, gap_at, noise_mm=args.noise)
    seam_end = SEAM_START + np.array([0.0, 0.0, SEAM_LENGTH])
    track = track_seam(cloud, SEAM_START, seam_end, stations=args.stations)
    path = track.smoothed_path()
    measured = track.measured

    W, H = 1420, 700
    image = Image.new("RGB", (W, H), (251, 252, 253))
    draw = ImageDraw.Draw(image)

    # -------------------------------------------------- left: the seam from above
    left, top, pw, ph = 70, 70, 560, 540
    lateral_span = max(6.0, args.bow * 2.2)

    def to_px(along_m, lateral_mm):
        x = left + (lateral_mm + lateral_span / 2) / lateral_span * pw
        y = top + ph - (along_m / SEAM_LENGTH) * ph
        return (x, y)

    draw.text((left, 34), "THE SEAM, LOOKING DOWN THE WELD", fill=INK)
    draw.rectangle([left, top, left + pw, top + ph], outline=(228, 232, 236))
    for mm in range(int(-lateral_span / 2), int(lateral_span / 2) + 1, 2):
        x = to_px(0, mm)[0]
        draw.line([(x, top), (x, top + ph)], fill=(240, 243, 246))
        draw.text((x - 6, top + ph + 6), f"{mm:+d}", fill=MUTED)
    draw.text((left + pw / 2 - 40, top + ph + 24), "millimetres across", fill=MUTED)

    draw.line([to_px(0, 0), to_px(SEAM_LENGTH, 0)], fill=TAUGHT, width=3)
    draw.text(to_px(SEAM_LENGTH, 0.4), "  taught path", fill=TAUGHT)

    actual = [to_px(z, bow * 1000) for z, bow in truth]
    draw.line(actual, fill=ACTUAL, width=3)
    draw.text(to_px(SEAM_LENGTH * 0.52, args.bow + 0.5), "  the real seam", fill=ACTUAL)

    for station in measured:
        px = to_px(station.along_m, station.position[0] * 1000)
        draw.ellipse([px[0] - 4, px[1] - 4, px[0] + 4, px[1] + 4], fill=MEASURED)
    if len(path) > 1:
        draw.line(
            [to_px(s.along_m, p[0] * 1000) for s, p in zip(measured, path, strict=False)],
            fill=MEASURED,
            width=2,
        )
    draw.text(to_px(SEAM_LENGTH * 0.18, -lateral_span / 2 + 0.6), "measured", fill=MEASURED)

    # ------------------------------------------------------- right: the gap profile
    gleft, gw = 760, 560
    gaps = track.gaps_mm()
    gmax = max(3.0, float(gaps.max()) * 1.2) if len(gaps) else 3.0

    draw.text((gleft, 34), "GAP BETWEEN THE FOLDED EDGES", fill=INK)
    draw.rectangle([gleft, top, gleft + gw, top + ph], outline=(228, 232, 236))

    def gap_px(along_m, gap_mm):
        return (gleft + gap_mm / gmax * gw, top + ph - (along_m / SEAM_LENGTH) * ph)

    touch_x = gap_px(0, TOUCHING_TOLERANCE_MM)[0]
    draw.line([(touch_x, top), (touch_x, top + ph)], fill=GOOD, width=1)
    draw.text((touch_x + 4, top + 4), "touching", fill=GOOD)

    for mm in range(0, int(gmax) + 1):
        x = gap_px(0, mm)[0]
        draw.line([(x, top), (x, top + ph)], fill=(243, 245, 247))
        draw.text((x - 4, top + ph + 6), str(mm), fill=MUTED)
    draw.text((gleft + gw / 2 - 30, top + ph + 24), "millimetres", fill=MUTED)

    if measured:
        draw.line(
            [gap_px(s.along_m, s.gap_mm) for s in measured],
            fill=BAD,
            width=3,
        )
        for station in measured:
            px = gap_px(station.along_m, station.gap_mm)
            colour = GOOD if station.touching else BAD
            draw.ellipse([px[0] - 4, px[1] - 4, px[0] + 4, px[1] + 4], fill=colour)

    summary = track.summary()
    lines = [
        f"seam length {summary['seam_length_mm']:.0f} mm, "
        f"{summary['stations_measured']} of {summary['of']} stations measured",
        f"the seam wanders {summary['path_wander_mm']:.2f} mm off the taught line, "
        f"which is what a fixed program welds into",
        f"gap: mean {summary['gap_mm_mean']:.2f} mm, worst {summary['gap_mm_max']:.2f} mm "
        f"at {summary['gap_worst_at_mm']:.0f} mm along, "
        f"{summary['stations_touching']} stations touching",
        f"sensor noise {args.noise:.1f} mm",
    ]
    for i, line in enumerate(lines):
        draw.text((70, H - 78 + i * 16), line, fill=INK if i < 3 else MUTED)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out)
    print("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
