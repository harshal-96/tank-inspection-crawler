"""A simulated ultrasonic thickness gauge: the probe's readings, from wave physics.

    python tank/ultrasound.py                  # gauge all 168 probe places, check against the manifest
    python tank/ultrasound.py --film DIR       # also write the wavefield frames and traces the film shows

The camera sees the inner face of the shell. Wall lost from the outer face changes nothing
the camera can see, and a contact probe finds it by timing sound. A pulse goes into the
steel, bounces off the back face and comes back, and keeps bouncing; the gap between one
back-wall echo and the next is the round trip, and the thickness is the sound velocity times
that gap over two. Timing echo to echo is what lets a gauge ignore the couplant under the
probe, whose thickness nobody controls.

Until now the survey's probe readings were the thickness model's own numbers (thickness.py),
looked up at the probe place. This runs the measurement instead: a finite-difference
simulation of the pressure wave through the wall, one dimension for the gauge and two for
the picture, read the way a gauge reads it. The gauge is calibrated the way a real one is,
on two reference blocks of known thickness, and then it never sees the answer: it is
handed only a wall, and the thickness it reports is compared with the manifest afterwards.

Tanks are often lined inside, and the lining sits between the probe and the steel. It is
why the gauge times echo to echo and not pulse to first echo: a gauge calibrated on bare
blocks that times the first echo counts the lining's transit as steel, about 1.17 mm extra
through 0.5 mm of epoxy, and would pass 15 of the 28 failing places. Echo to echo cancels
the lining, but only once the lining has stopped ringing: its own echo is the loudest thing
after the pulse, and an autocorrelation over the whole trace locks onto it. The gauge opens
its window four lining round trips in, when the ringing is under the noise.

What it leaves out, said plainly. Longitudinal waves only, so no shear waves and no mode
conversion. A flat back face under the probe: the patches are 0.55 to 1.2 m across and
the probe is 10 mm, so the slope under it is under a degree. No attenuation, a velocity
that is the same everywhere in the shell, and electronic noise at a fixed level. A real
shell's velocity varies by a percent or so with the steel, which is why a real gauge is
calibrated on a block cut from the same plate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "tank" / "data" / "survey" / "manifest.json"
OUT = ROOT / "tank" / "data" / "ultrasound.json"

C_STEEL, RHO_STEEL = 5920.0, 7850.0      # longitudinal velocity of carbon steel, m/s; density, kg/m3
C_GEL, RHO_GEL = 1500.0, 1000.0          # water-based couplant gel
F0 = 5.0e6                               # a 5 MHz contact probe, the usual choice for plate
GEL_MM = 0.06                            # couplant film under the probe; echo-to-echo timing ignores it
FS = 100.0e6                             # the gauge digitises at 100 MHz
NOISE = 0.01                             # electronic noise, as a share of the first back-wall echo
BLANK_S = 0.8e-6                         # the transmit pulse rings this long; nothing is read inside it
DISPLAY_MM = 0.01                        # the gauge shows two decimals
LIMIT = 0.6                              # the survey's screening limit, a share of nominal
SPREAD = 0.005                           # plate-to-plate velocity spread in carbon steel, about half a percent
PLATES = 8                               # plates per course, as the scene builds the shell
# an epoxy lining on the inner face, between the probe and the steel. 2540 m/s is the epoxy resin
# row of the gauge makers' velocity tables; 1200 kg/m3 puts it at 3.0 MRayl, inside the 0.88 to
# 6.25 MRayl measured across epoxies (J. Acoust. Soc. Am. 128, 2010, doi:10.1121/1.3483734).
# The density sets only how much sound crosses into the steel, not when it comes back.
C_LINING, RHO_LINING = 2540.0, 1200.0
LINING_MM = 0.5                          # 20 mils, where API RP 652 splits thin-film from thick-film linings
LININGS_MM = (0.25, 0.5, 1.0)


def pulse(t, t0=0.4e-6, tau=0.12e-6):
    """A Gaussian-windowed 5 MHz burst, about two and a half cycles, like a damped contact probe."""
    return np.sin(2 * np.pi * F0 * (t - t0)) * np.exp(-(((t - t0) / tau) ** 2))


# ------------------------------------------------------------------ one dimension: the gauge
def trace_1d(thick_mm: float, t_end=14e-6, dx_target=2.0e-5, c_steel=C_STEEL, lining_mm=0.0):
    """Pressure at the probe face for a wall of this thickness, sampled at the gauge's rate.

    Staggered velocity and pressure on a line through the probe, the couplant, any lining and
    the wall, with the outer face of the wall a free surface (steel against air) and the far
    side of the probe an absorbing layer, which is what the probe's backing does.
    """
    n_steel = max(8, round(thick_mm * 1e-3 / dx_target))
    dx = thick_mm * 1e-3 / n_steel            # the wall is a whole number of cells, so its thickness is exact
    n_sponge, n_gel = 260, max(2, round(GEL_MM * 1e-3 / dx))
    n_lin = round(lining_mm * 1e-3 / dx)
    src = n_sponge + 4
    n = src + n_gel + n_lin + n_steel + 1     # the last pressure node sits on the back face, held at zero
    c = np.full(n, C_GEL)
    rho = np.full(n, RHO_GEL)
    c[src + n_gel:src + n_gel + n_lin] = C_LINING
    rho[src + n_gel:src + n_gel + n_lin] = RHO_LINING
    c[src + n_gel + n_lin:] = c_steel
    rho[src + n_gel + n_lin:] = RHO_STEEL
    k = rho * c ** 2                          # bulk modulus at each pressure node
    rho_f = 0.5 * (rho[1:] + rho[:-1])        # density at the velocity nodes between them
    damp = np.ones(n)
    ramp = np.linspace(1.0, 0.0, n_sponge) ** 2
    damp[:n_sponge] = np.exp(-0.18 * ramp)
    dt = 0.9 * dx / max(c_steel, C_STEEL)
    steps = int(t_end / dt)
    p = np.zeros(n)
    v = np.zeros(n - 1)
    rec = np.empty(steps)
    for s in range(steps):
        v -= dt / (rho_f * dx) * (p[1:] - p[:-1])
        v *= damp[1:]
        p[1:-1] -= dt * k[1:-1] / dx * (v[1:] - v[:-1])
        p[src] += pulse(s * dt) * 0.05
        p *= damp
        p[-1] = 0.0
        rec[s] = p[src]
    t = np.arange(steps) * dt
    ts = np.arange(0, t_end, 1 / FS)
    return ts, np.interp(ts, t, rec)


def envelope(x):
    """The magnitude of the analytic signal, by FFT, so no filter design is needed."""
    n = len(x)
    f = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1
    h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    return np.abs(np.fft.ifft(f * h))


def round_trip(ts, x, start_s=BLANK_S):
    """Echo-to-echo time: the lag that best lines the echo train up with itself.

    Coarse from the envelope, then refined on the waveform itself with a parabola through
    the correlation peak, which is how a gauge gets below one sample. Only the trace after
    start_s is read: the transmit pulse's ringing, and through a lining, the lining's own.
    """
    keep = ts >= start_s
    y = x[keep] - x[keep].mean()
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    env = envelope(np.concatenate([ac, ac[::-1]]))[:len(ac)]
    lo = int(0.45e-6 * FS)                    # the pulse's own width; a real lag is longer
    i = lo + int(np.argmax(env[lo:]))
    j0, j1 = max(lo, i - 12), i + 13
    j = j0 + int(np.argmax(ac[j0:j1]))
    a, b, c = ac[j - 1], ac[j], ac[j + 1]
    frac = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return (j + frac) / FS


class Gauge:
    """Two-point calibration on reference blocks, then readings it is never told the answer to."""

    def __init__(self, blocks=(5.0, 10.0), rng=None):
        self.rng = rng or np.random.default_rng(0)
        rt = [round_trip(*self.listen(b)) for b in blocks]
        # thickness = a * round_trip + b, fitted on the blocks: velocity and zero offset
        self.a = (blocks[1] - blocks[0]) / (rt[1] - rt[0])
        self.b = blocks[0] - self.a * rt[0]
        self.velocity = 2 * self.a / 1000.0      # m/s, as the gauge would display it

    def listen(self, thick_mm, c_steel=C_STEEL, lining_mm=0.0):
        ts, x = trace_1d(thick_mm, c_steel=c_steel, lining_mm=lining_mm)
        b1 = np.abs(x[ts > BLANK_S]).max()
        return ts, x + self.rng.normal(0, NOISE * b1, len(x))

    def read(self, thick_mm, c_steel=C_STEEL, lining_mm=0.0, start_s=BLANK_S):
        ts, x = self.listen(thick_mm, c_steel, lining_mm)
        raw = self.a * round_trip(ts, x, start_s) + self.b
        return round(raw / DISPLAY_MM) * DISPLAY_MM, ts, x


def past_the_lining(lining_mm, c_lining=C_LINING, rho_lining=RHO_LINING):
    """When the lining has stopped ringing: the read window a gauge opens before timing echo to echo.

    Sound trapped in the lining loses the share that leaks out at each face on every round
    trip, R(lining to gel) * R(lining to steel), about 0.3 here; the window opens once that
    ringing is under the noise. An operator sets the same gate by eye on the A-scan, where
    the lining's echo is the first and largest thing after the pulse.
    """
    z_g, z_l, z_s = RHO_GEL * C_GEL, rho_lining * c_lining, RHO_STEEL * C_STEEL
    keep = abs((z_g - z_l) / (z_g + z_l) * (z_s - z_l) / (z_s + z_l))
    trips = math.ceil(math.log(NOISE) / math.log(keep))
    return BLANK_S + trips * 2 * lining_mm * 1e-3 / c_lining, trips


def first_echo_mm(true_mm, lining_mm, c_lining=C_LINING):
    """What a gauge that times the pulse to the first back-wall echo reads through a lining.

    Calibrated on bare blocks, it takes every microsecond after the pulse for steel, so the
    lining's own round trip, 2L/c_lining, comes back as c_steel * L / c_lining of extra wall.
    That is arithmetic, not simulation, and it is the best case: it assumes the gauge picks
    the right echo, when the lining's own echo arrives first and is the stronger of the two.
    """
    return true_mm + C_STEEL * lining_mm / c_lining


def lining_study(gauge) -> dict:
    """Through a lining, does timing echo to echo still read the steel, and what would the first echo read?"""
    walls = (3.86, 6.0, 10.0)
    sims = []
    for lin in LININGS_MM:
        start, trips = past_the_lining(lin)
        for t in walls:
            whole, _, _ = gauge.read(t, lining_mm=lin)
            gated, _, _ = gauge.read(t, lining_mm=lin, start_s=start)
            sims.append({"lining_mm": lin, "true_mm": t, "window_opens_us": round(start * 1e6, 2),
                         "lining_round_trips_waited": trips, "echo_to_echo_whole_trace_mm": round(whole, 2),
                         "echo_to_echo_mm": round(gated, 2), "first_echo_mm": round(first_echo_mm(t, lin), 2)})
    rows = [c for _, c in places()]
    fails = [c for c in rows if c["true_thickness_mm"] <= LIMIT * c["nominal_mm"]]
    passed = [c for c in fails if first_echo_mm(c["true_thickness_mm"], LINING_MM) > LIMIT * c["nominal_mm"]]
    # every probe place again, lined and with each plate's own velocity: the survey as it would be read
    start, _ = past_the_lining(LINING_MM)
    err, wrong, failed = 0.0, 0, 0
    for c in rows:
        t, lim = c["true_thickness_mm"], LIMIT * c["nominal_mm"]
        got, _, _ = gauge.read(t, plate_velocity(c), LINING_MM, start)
        err = max(err, abs(got - t))
        failed += got <= lim
        wrong += (got <= lim) != (t <= lim)
    return {
        "lining": {"c_m_s": C_LINING, "rho_kg_m3": RHO_LINING, "impedance_mrayl": round(C_LINING * RHO_LINING / 1e6, 2)},
        "simulated": sims,
        "echo_to_echo_max_abs_error_mm": round(max(abs(s["echo_to_echo_mm"] - s["true_mm"]) for s in sims), 3),
        "whole_trace_max_abs_error_mm": round(max(abs(s["echo_to_echo_whole_trace_mm"] - s["true_mm"]) for s in sims), 3),
        "first_echo_extra_mm_at": {str(lin): round(C_STEEL * lin / C_LINING, 3) for lin in LININGS_MM},
        "at_lining_mm": LINING_MM,
        "fail_by_manifest": len(fails),
        "first_echo_would_pass": len(passed),
        "every_place_lined_with_velocity_spread": {"places": len(rows), "max_abs_error_mm": round(err, 3),
                                                   "fail_by_gauge": failed, "verdicts_that_differ": wrong},
    }


def convergence() -> list:
    """Halve the grid and watch the simulation's own error: a 10 mm wall timed raw, before calibration.

    The wall comes back long by about half a cell, where the staggered grid puts the free
    back face. That is a fixed length, not a velocity error, so the two-block calibration's
    zero term takes it out, which is why the calibrated gauge lands within 0.005 mm.
    """
    out = []
    for dx in (4e-5, 2e-5, 1e-5):
        ts, x = trace_1d(10.0, dx_target=dx)
        seen_mm = C_STEEL * round_trip(ts, x) / 2 * 1e3
        out.append({"dx_um": round(dx * 1e6), "cells_per_wavelength": round(C_STEEL / F0 / dx),
                    "wall_long_by_um": round(float(seen_mm - 10.0) * 1e3, 1), "half_cell_um": round(dx * 1e6 / 2, 1)})
    return out


# ------------------------------------------------------------------ two dimensions: the picture
def wavefield_2d(thick_mm, width_mm=30.0, depth_mm=12.0, aperture_mm=8.0, t_end=9.0e-6,
                 frames=150, dx=5.0e-5):
    """The pressure field in a slice through the wall under the probe, frames evenly in time.

    Steel between a free inner face and a free outer face (air on both sides, the probe
    pressed on the inner face), absorbing at the two cut edges of the slice.
    """
    nx, nz = round(width_mm * 1e-3 / dx), round(depth_mm * 1e-3 / dx)
    zt = round(thick_mm * 1e-3 / dx)
    steel = np.zeros((nz, nx), bool)
    steel[1:zt, :] = True
    k = RHO_STEEL * C_STEEL ** 2
    dt = 0.9 * dx / (C_STEEL * math.sqrt(2))
    steps = int(t_end / dt)
    every = max(1, steps // frames)
    sp = 50
    damp = np.ones(nx)
    damp[:sp] = np.exp(-0.12 * np.linspace(1, 0, sp) ** 2)
    damp[-sp:] = damp[:sp][::-1]
    damp = damp[None, :]
    p = np.zeros((nz, nx))
    vx = np.zeros((nz, nx - 1))
    vz = np.zeros((nz - 1, nx))
    a0 = round((width_mm - aperture_mm) / 2 * 1e-3 / dx)
    a1 = a0 + round(aperture_mm * 1e-3 / dx)
    out, times = [], []
    for s in range(steps):
        vx -= dt / (RHO_STEEL * dx) * (p[:, 1:] - p[:, :-1])
        vz -= dt / (RHO_STEEL * dx) * (p[1:, :] - p[:-1, :])
        vx *= damp[:, 1:]
        p[1:-1, 1:-1] -= dt * k / dx * ((vx[1:-1, 1:] - vx[1:-1, :-1]) + (vz[1:, 1:-1] - vz[:-1, 1:-1]))
        p[1, a0:a1] += pulse(s * dt) * 0.05
        p *= damp
        p[~steel] = 0.0
        if s % every == 0 and len(out) < frames:
            out.append(p.copy())
            times.append(s * dt)
    return np.array(out), np.array(times), steel


def paint(field, steel, scale, w_px=1120):
    """Pressure to colour: warm for compression, cool for rarefaction, on steel grey; air near black."""
    base = np.array([0.155, 0.17, 0.195])
    warm = np.array([1.0, 0.60, 0.30])
    cool = np.array([0.61, 0.76, 1.0])
    air = np.array([0.045, 0.05, 0.06])
    q = np.clip(field / scale, -1, 1)
    m = np.abs(q) ** 0.55
    rgb = base[None, None] * (1 - m[..., None])
    rgb += np.where(q[..., None] > 0, warm, cool) * m[..., None]
    rgb[~steel] = air
    img = (np.clip(rgb, 0, 1) ** (1 / 1.0) * 255).astype(np.uint8)
    from PIL import Image
    h_px = round(w_px * steel.shape[0] / steel.shape[1])
    return Image.fromarray(img).resize((w_px, h_px), Image.BILINEAR)


# ------------------------------------------------------------------ the survey
def places():
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for f in m["frames"]:
        for c in f["probe_candidates"]:
            yield f["station"], c


def plate_velocity(c: dict) -> float:
    """Each plate's own sound velocity, within SPREAD of the book value, fixed by which plate it is.

    The gauge is calibrated on blocks at the book value and is not told this; it is the main
    reason a real reading is off by a few hundredths of a millimetre per centimetre of wall.
    """
    course = int(c["z_m"] // 2.0)
    plate = int((c["theta_rad"] % math.tau) // (math.tau / PLATES))
    u = np.random.default_rng(1000 * course + plate).uniform(-1, 1)
    return C_STEEL * (1 + SPREAD * u)


def run(film_dir: Path | None = None) -> dict:
    gauge = Gauge()
    rows = []
    for station, c in places():
        t = c["true_thickness_mm"]
        lim = LIMIT * c["nominal_mm"]
        ideal, _, _ = gauge.read(t)
        cv = plate_velocity(c)
        real, _, _ = gauge.read(t, cv)
        rows.append({"station": station, "offset_m": c["offset_m"], "nominal_mm": c["nominal_mm"],
                     "true_mm": t, "gauge_mm": round(ideal, 2), "plate_velocity_m_s": round(cv, 1),
                     "gauge_spread_mm": round(real, 2), "fails_true": t <= lim,
                     "fails_gauge": ideal <= lim, "fails_spread": real <= lim})
    err = np.array([r["gauge_mm"] - r["true_mm"] for r in rows])
    err_s = np.array([r["gauge_spread_mm"] - r["true_mm"] for r in rows])
    worst = min(rows, key=lambda r: r["true_mm"] / r["nominal_mm"])
    summary = {
        "what_this_is": "a simulated pulse-echo thickness gauge read at every probe place in the survey",
        "physics": {"c_steel_m_s": C_STEEL, "probe_mhz": F0 / 1e6, "digitiser_mhz": FS / 1e6,
                    "couplant_mm": GEL_MM, "noise_share_of_first_echo": NOISE,
                    "method": "1D finite-difference pressure wave, echo-to-echo timing by autocorrelation"},
        "calibration": {"blocks_mm": [5.0, 10.0], "velocity_m_s": round(gauge.velocity, 1)},
        "places": len(rows),
        "max_abs_error_mm": round(float(np.abs(err).max()), 3),
        "mean_error_mm": round(float(err.mean()), 4),
        "fail_by_manifest": sum(r["fails_true"] for r in rows),
        "fail_by_gauge": sum(r["fails_gauge"] for r in rows),
        "verdicts_that_differ": sum(r["fails_true"] != r["fails_gauge"] for r in rows),
        "with_velocity_spread": {
            "spread": SPREAD, "max_abs_error_mm": round(float(np.abs(err_s).max()), 3),
            "fail_by_gauge": sum(r["fails_spread"] for r in rows),
            "verdicts_that_differ": sum(r["fails_true"] != r["fails_spread"] for r in rows),
            "closest_to_the_limit_mm": round(float(min(abs(r["true_mm"] - LIMIT * r["nominal_mm"]) for r in rows)), 3)},
        "worst_place": worst,
        "through_a_lining": lining_study(gauge),
        "grid_convergence": convergence(),
        "readings": rows,
    }
    if film_dir:
        film(film_dir, gauge, worst, summary)
    return summary


def film(d: Path, gauge, worst, summary):
    """The two walls the film compares, sound and thinned, as frames and traces."""
    d.mkdir(parents=True, exist_ok=True)
    walls = [("full", worst["nominal_mm"]), ("thin", worst["true_mm"])]
    fields = {}
    for name, t in walls:
        f, times, steel = wavefield_2d(t)
        fields[name] = (f, times, steel)
    scale = np.percentile(np.abs(fields["full"][0][5:20]), 99.7)
    traces = {}
    for name, t in walls:
        f, times, steel = fields[name]
        for i, fr in enumerate(f):
            paint(fr, steel, scale).save(d / f"{name}_{i:04d}.jpg", quality=90)
        reading, ts, x = gauge.read(t)
        keep = ts <= times[-1]
        traces[name] = {"thick_mm": t, "reading_mm": reading, "round_trip_us": round(2 * t * 1e-3 / C_STEEL * 1e6, 3),
                        "t_us": [round(v * 1e6, 3) for v in ts[keep]], "a": [round(float(v), 4) for v in x[keep]]}
    (d / "ultrasound_film.json").write_text(json.dumps({
        "frames": len(fields["full"][0]), "t_end_us": round(float(fields["full"][1][-1] * 1e6), 3),
        "worst": {k: worst[k] for k in ("station", "offset_m", "nominal_mm", "true_mm", "gauge_mm")},
        "places": summary["places"], "max_abs_error_mm": summary["max_abs_error_mm"],
        "fail_by_gauge": summary["fail_by_gauge"], "verdicts_that_differ": summary["verdicts_that_differ"],
        "with_velocity_spread": summary["with_velocity_spread"],
        "traces": traces}), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--film", type=Path, help="write wavefield frames and traces here")
    ap.add_argument("--no-write", action="store_true", help="print only")
    a = ap.parse_args()
    s = run(a.film)
    if not a.no_write:
        OUT.write_text(json.dumps(s, indent=1), encoding="utf-8")
    w = s["worst"] if "worst" in s else s["worst_place"]
    print(f"gauge calibrated on 5 and 10 mm blocks: velocity {s['calibration']['velocity_m_s']} m/s")
    print(f"{s['places']} places gauged; largest error {s['max_abs_error_mm']} mm, mean {s['mean_error_mm']} mm")
    print(f"below {LIMIT:.0%} of nominal: {s['fail_by_manifest']} by the manifest, {s['fail_by_gauge']} by the gauge; "
          f"{s['verdicts_that_differ']} verdicts differ")
    v = s["with_velocity_spread"]
    print(f"with a {v['spread']:.1%} plate-to-plate velocity spread the gauge is not told about: largest error "
          f"{v['max_abs_error_mm']} mm, {v['fail_by_gauge']} below the limit, {v['verdicts_that_differ']} verdicts differ "
          f"(the closest place sits {v['closest_to_the_limit_mm']} mm from the limit)")
    print(f"thinnest place: station {w['station']} at {w['offset_m']:+.2f} m, {w['true_mm']} mm of {w['nominal_mm']}, "
          f"gauge reads {w['gauge_mm']}")
    L = s["through_a_lining"]
    print(f"through epoxy linings of {', '.join(str(x) for x in LININGS_MM)} mm: echo to echo over the whole trace is off "
          f"by up to {L['whole_trace_max_abs_error_mm']} mm (it locks onto the lining's own ringing); with the window opened "
          f"past it, {'every reading matches to the last digit' if L['echo_to_echo_max_abs_error_mm'] < DISPLAY_MM else 'off by ' + str(L['echo_to_echo_max_abs_error_mm']) + ' mm'}")
    print(f"timed pulse to first echo through {L['at_lining_mm']} mm, the gauge adds {L['first_echo_extra_mm_at'][str(L['at_lining_mm'])]} mm "
          f"and would pass {L['first_echo_would_pass']} of the {L['fail_by_manifest']} failing places")
    E = L["every_place_lined_with_velocity_spread"]
    print(f"all {E['places']} places through {L['at_lining_mm']} mm of epoxy, each plate at its own velocity: largest error "
          f"{E['max_abs_error_mm']} mm, {E['fail_by_gauge']} below the limit, {E['verdicts_that_differ']} verdicts differ")
    print("grid convergence, 10 mm wall timed raw: " + ", ".join(
        f"{g['dx_um']} um grid reads {g['wall_long_by_um']} um long (half a cell {g['half_cell_um']})" for g in s["grid_convergence"]))


if __name__ == "__main__":
    main()
