"""Draw what the simulated gauge hears through a bare wall and through a lined one.

Two A-scans of the same 6 mm wall, from tank/ultrasound.py, with the readings the run
reports for each way of timing them. Nothing here is drawn by hand: the traces are the
simulation's, and the numbers on them are the gauge's.

    python scripts/draw_ultrasound.py --out docs/figures/ultrasound_lining.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tank import ultrasound as us  # noqa: E402

INK = (28, 32, 38)
MUTED = (120, 128, 140)
FAINT = (226, 230, 236)
GATE = (240, 242, 246)
SOUND = (47, 111, 176)
LINED = (214, 110, 40)
WRONG = (200, 62, 62)

W, H = 1400, 860
LEFT, RIGHT = 90, 1360
T_END_US = 14.0
WALL = 6.0


def font(size: int, bold: bool = False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), ("arialbd.ttf" if bold else "arial.ttf"),
                 ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def x_of(t_us: float) -> float:
    return LEFT + (RIGHT - LEFT) * t_us / T_END_US


def scan(d: ImageDraw.ImageDraw, top: int, height: int, ts, x, colour, gate_us: float, echo_scale: float,
         marks: list[tuple[float, str]]):
    """One A-scan: the shaded part is what the gauge does not read."""
    mid = top + height / 2
    d.rectangle([x_of(0), top, x_of(gate_us), top + height], fill=GATE)
    d.line([LEFT, mid, RIGHT, mid], fill=FAINT, width=1)
    for t in range(0, int(T_END_US) + 1, 2):
        d.line([x_of(t), top + height, x_of(t), top + height + 6], fill=MUTED, width=1)
        d.text((x_of(t) - 8, top + height + 9), f"{t} µs", fill=MUTED, font=font(15))
    y = np.clip(x / echo_scale, -1, 1)
    pts = [(x_of(t * 1e6), mid - v * (height / 2 - 6)) for t, v in zip(ts, y) if t * 1e6 <= T_END_US]
    d.line(pts, fill=colour, width=2, joint="curve")
    for t, label in marks:
        d.text((x_of(t) - 9, top - 22), label, fill=INK, font=font(15, True))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("docs/figures/ultrasound_lining.png"))
    a = ap.parse_args()

    gauge = us.Gauge()
    rt_us = 2 * WALL * 1e-3 / us.C_STEEL * 1e6
    lining_us = 2 * us.LINING_MM * 1e-3 / us.C_LINING * 1e6
    start_s, trips = us.past_the_lining(us.LINING_MM)
    bare, ts0, x0 = gauge.read(WALL)
    whole, ts1, x1 = gauge.read(WALL, lining_mm=us.LINING_MM)
    gated, _, _ = gauge.read(WALL, lining_mm=us.LINING_MM, start_s=start_s)
    first = us.first_echo_mm(WALL, us.LINING_MM)

    def echo(ts, x, t_us):
        m = (ts >= (t_us - 0.15) * 1e-6) & (ts <= (t_us + 0.15) * 1e-6)
        return float(np.abs(x[m]).max())

    t0 = 0.4                                   # the pulse's centre, us.pulse's t0
    b1_bare, b1_lined = t0 + rt_us, t0 + lining_us + rt_us
    loud = echo(ts1, x1, t0 + lining_us) / echo(ts1, x1, b1_lined)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((LEFT, 26), f"The same {WALL:.0f} mm wall, bare and under {us.LINING_MM} mm of epoxy, as the simulated gauge hears it",
           fill=INK, font=font(26, True))
    d.text((LEFT, 64), "Pressure at the probe face. Shaded: before the gauge's read window opens. "
           "The transmit pulse and the lining's echo run off the scale.", fill=MUTED, font=font(17))

    top1, top2, h = 150, 470, 190
    d.text((LEFT, top1 - 52), "Bare wall", fill=SOUND, font=font(19, True))
    scan(d, top1, h, ts0, x0, SOUND, us.BLANK_S * 1e6, 1.6 * echo(ts0, x0, b1_bare),
         [(b1_bare + k * rt_us, f"B{k + 1}") for k in range(6) if b1_bare + k * rt_us < T_END_US - 0.3])
    d.text((LEFT, top1 + h + 36), f"Echo to echo, {rt_us:.2f} µs apart:  reads {bare:.2f} mm", fill=INK, font=font(18, True))

    d.text((LEFT, top2 - 52), f"Under {us.LINING_MM} mm of epoxy", fill=LINED, font=font(19, True))
    scan(d, top2, h, ts1, x1, LINED, start_s * 1e6, 1.6 * echo(ts1, x1, b1_lined),
         [(b1_lined + k * rt_us, f"B{k + 1}") for k in range(6) if b1_lined + k * rt_us < T_END_US - 0.3])
    d.text((x_of(t0 + lining_us) - 9, top2 - 22), f"lining echo, {loud:.1f}× B1", fill=INK, font=font(15, True))
    d.text((x_of(start_s * 1e6) + 6, top2 + h - 24), f"window opens {start_s * 1e6:.2f} µs, {trips} lining round trips in",
           fill=MUTED, font=font(15))
    y = top2 + h + 36
    rows = [(f"Echo to echo, window past the lining's ringing:  reads {gated:.2f} mm", INK),
            (f"Echo to echo over the whole trace, locked onto the lining's ringing:  reads {whole:.2f} mm", WRONG),
            (f"Pulse to first echo, calibrated on bare blocks, even picking the right echo:  reads {first:.2f} mm", WRONG)]
    for text, colour in rows:
        d.text((LEFT, y), text, fill=colour, font=font(18, True))
        y += 30
    d.text((LEFT, H - 34), f"tank/ultrasound.py · 1D finite-difference pressure wave · {us.F0 / 1e6:.0f} MHz · steel "
           f"{us.C_STEEL:.0f} m/s · epoxy {us.C_LINING:.0f} m/s, {us.C_LINING * us.RHO_LINING / 1e6:.1f} MRayl · "
           f"{us.NOISE:.0%} noise", fill=MUTED, font=font(15))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out, optimize=True)
    print(f"wrote {a.out}: bare {bare:.2f}, lined gated {gated:.2f}, whole trace {whole:.2f}, first echo {first:.2f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
