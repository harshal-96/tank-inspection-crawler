"""What holds the crawler on the wall, and where that stops being true.

    python tank/adhesion.py
    python tank/adhesion.py --mass 8.0 --friction 0.25 --wet-friction 0.15 --gap-mm 1.0

The simulation places the crawler on the shell with a matrix. There is no
physics in it, no mass and no adhesion, and a rendered robot sticks to a
rendered wall for the same reason a painting of a robot does. So the question
"what holds it up" is not answered anywhere in the renderer, and it is the
first thing anybody who has been inside a tank will ask.

This answers it the way it would be answered on a drawing: a load case, a
required holding force, and the margin left over. Nothing here is simulated.
It is arithmetic against the geometry the scene is actually built from.

The principle. An API 650 shell is carbon steel, which is ferromagnetic, which
is why magnetic crawlers exist at all. Four wheels carry permanent magnets and
the holding force is passive. That last word matters more than any number
below: an electromagnet that drops the robot when a fuse goes is not a safety
case, and every serious crawler uses permanent magnets so that losing power
makes it stop rather than fall.

Three things have to be true at once.

  sliding   friction has to carry the weight. The magnets press the wheels
            into the plate, friction acts along it, and the robot slides down
            the wall when the weight beats it.
  peel      the centre of mass stands off the wall, so gravity tries to rotate
            the robot off its upper wheels. That is a smaller force than the
            sliding case here but it acts on two wheels instead of four.
  tether    the umbilical hangs below the robot and pulls down. It is small
            per metre and it is not small at twelve metres.

And three things quietly destroy all of it.

  liftoff   holding force falls roughly with the square of the gap. Paint,
            mill scale and rust are gap. A millimetre of corrosion product can
            halve the grip, and the places worth inspecting are exactly the
            places that have it.
  thin plate a magnetic circuit needs steel to carry its flux. Below roughly
            six to eight millimetres the plate saturates and the magnet cannot
            develop its rated pull however good it is. This tank steps from
            12 mm at the bottom course to 6 mm at the top, so the grip is
            weakest at the top, which is also furthest to fall.
  water     the probe's couplant is water, pumped on to the wall the crawler
            is standing on, and a wet tyre does not grip like a dry one. So
            every case is worked twice, dry and wet. The wet friction is an
            assumption, not a measured or quoted figure, and it is the first
            number a test on real plate should replace.

Reading the table. The margin is over the safety factor: below 1.0 the
crawler is holding with less than the factor in hand ("below SF"). The actual
factor is the margin times the safety factor; below 1.0 it is not holding at
all, and the flag says how it lets go (SLIDES or PEELS).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

G = 9.81


def scene_constants(*names) -> dict:
    """Read named constants out of the builder without importing it.

    tank/sim/blender_tank.py imports bpy and will not load outside Blender, and
    copying its numbers here would mean two places to change the wheelbase and
    one of them getting forgotten. So the source is parsed for the literal
    assignments instead, which is ugly and cannot go stale.
    """
    import ast
    source = (ROOT / "tank" / "sim" / "blender_tank.py").read_text(encoding="utf-8")
    wanted, out = set(names), {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                with contextlib.suppress(ValueError):
                    out[target.id] = ast.literal_eval(node.value)
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"not found in blender_tank.py: {sorted(missing)}")
    return out

# Itemised rather than guessed as one number, so a reader can argue with a line
# instead of with the total. Masses in kg, for the crawler as modelled.
MASS_BUDGET = (
    ("deck and frame, 3 mm aluminium", 1.55),
    ("body shell and lid, 2 mm aluminium", 1.80),
    ("four magnetic wheel assemblies", 1.40),
    ("drive motors and gearboxes", 0.70),
    ("camera head, lamps, probe and arm", 0.75),
    ("battery, controller and wiring", 0.60),
)

TETHER_KG_PER_M = 0.09
TETHER_HANG_M = 6.0        # how much of it is hanging below the robot

# Centre of mass stands off the plate by about the body's mid height.
COM_STANDOFF_M = 0.16

# What a decent magnetic wheel the size of the modelled one develops on thick,
# clean, bare steel. The scene's wheel is 156 mm across and 52 mm wide
# (WHEEL_R 0.078 is its radius, WHEEL_WIDTH 0.052 its width). An assumed
# figure, to be replaced by the chosen wheel's datasheet pull and derated
# below, never a result; main() prints the rated pull the worst case needs, which
# is the number to look for on that datasheet.
WHEEL_RATED_N = 220.0

# The tyre with the probe's couplant water under it. An assumption, stated as
# one: water lowers rubber-on-steel friction, and the crawler pumps it on to
# the wall it stands on.
WET_FRICTION = 0.20

# bead_surface in tank/sim/meshkit.py puts each toe of the modelled bead 1.35
# bead radii from the centreline.
BEAD_HALF_WIDTH_PER_R = 1.35

# Where the rated figure stops being available.
FULL_FLUX_PLATE_MM = 8.0   # below this the plate starts to saturate
RATED_AT_GAP_MM = 0.2      # the gap the rating assumes: bare, clean steel

# The magnet's own effective gap: pole pieces, the tyre, and the distance the
# flux already crosses inside the wheel before it reaches the plate. It is what
# stops a coating being catastrophic, and leaving it out is the usual way this
# calculation gets written wrong. Modelled as inverse square from contact, a
# half millimetre of paint appears to remove ninety percent of the grip, which
# is not what magnetic wheels do on painted steel.
MAGNET_INTERNAL_GAP_MM = 2.0


def plate_derate(thickness_mm: float) -> float:
    """How much of the rated pull a plate this thin can actually carry."""
    if thickness_mm >= FULL_FLUX_PLATE_MM:
        return 1.0
    # Falls off with the steel available to carry the flux, with a floor,
    # because a thin plate still holds something.
    return max(0.45, thickness_mm / FULL_FLUX_PLATE_MM)


def gap_derate(gap_mm: float) -> float:
    """How much of it survives paint, scale and rust under the wheel.

    Falls with the square of the total magnetic gap, which is the coating plus
    the gap the wheel already has inside it. Normalised so the rated figure is
    what you get on bare steel.
    """
    def pull(extra_mm: float) -> float:
        return 1.0 / (MAGNET_INTERNAL_GAP_MM + extra_mm) ** 2

    return pull(max(gap_mm, 0.0)) / pull(RATED_AT_GAP_MM)


def budget(mass_kg: float, friction: float, safety: float, gap_mm: float,
           plate_mm: float, wheels: int = 4) -> dict:
    tether_n = TETHER_KG_PER_M * TETHER_HANG_M * G
    weight_n = mass_kg * G
    pulling_down = weight_n + tether_n

    # 1. Sliding. Friction has to carry everything hanging on the robot.
    need_total_sliding = safety * pulling_down / friction

    # 2. Peel. The weight acts through a centre of mass standing off the wall,
    # which levers the upper wheels away from it. Resisted across the track.
    wheel_at = scene_constants("WHEEL_AT")["WHEEL_AT"]
    track_m = max(abs(y) for _, y in wheel_at) * 2
    peel_moment = pulling_down * COM_STANDOFF_M
    peel_per_upper_wheel = safety * peel_moment / track_m / (wheels / 2)

    per_wheel_need = max(need_total_sliding / wheels, peel_per_upper_wheel)
    governs = ("sliding" if need_total_sliding / wheels >= peel_per_upper_wheel
               else "peel")

    available = WHEEL_RATED_N * plate_derate(plate_mm) * gap_derate(gap_mm)
    margin = available / per_wheel_need if per_wheel_need else math.inf
    return {
        "mass_kg": mass_kg,
        "weight_n": weight_n,
        "tether_n": tether_n,
        "track_m": track_m,
        "need_total_sliding_n": need_total_sliding,
        "peel_per_upper_wheel_n": peel_per_upper_wheel,
        "per_wheel_need_n": per_wheel_need,
        "per_wheel_available_n": available,
        "plate_factor": plate_derate(plate_mm),
        "gap_factor": gap_derate(gap_mm),
        "margin": margin,
        # Margin is over the safety factor; this is the factor actually left.
        "actual_factor": margin * safety,
        "governs": governs,
    }


def flag(got: dict) -> str:
    """Below the safety factor is not the same as falling off, so say which."""
    if got["actual_factor"] < 1.0:
        return "SLIDES" if got["governs"] == "sliding" else "PEELS"
    if got["margin"] < 1.0:
        return "below SF"
    return ""


def rated_pull_needed(got: dict) -> float:
    """The bare-steel rating a wheel would need for this case to hold the factor."""
    return got["per_wheel_need_n"] / (got["plate_factor"] * got["gap_factor"])


def largest_gap(mass_kg: float, friction: float, safety: float, plate_mm: float,
                factor: float) -> float:
    """The most liftoff this plate takes before the factor left drops below `factor`.

    Found by bisection on budget() rather than solved in closed form, so it
    stays right if the derating model changes. The search starts at
    RATED_AT_GAP_MM, because that is bare clean steel in this model and no
    wheel sits closer than that. Zero if it fails even there.
    """
    def holds(gap: float) -> bool:
        return budget(mass_kg, friction, safety, gap, plate_mm)["actual_factor"] >= factor

    if not holds(RATED_AT_GAP_MM):
        return 0.0
    lo, hi = RATED_AT_GAP_MM, 50.0
    if holds(hi):
        return math.inf
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if holds(mid) else (lo, mid)
    return lo


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float,
                        default=sum(m for _, m in MASS_BUDGET))
    parser.add_argument("--friction", type=float, default=0.35,
                        help="rubber tyred magnetic wheel on clean steel, dry")
    parser.add_argument("--wet-friction", type=float, default=WET_FRICTION,
                        help="the same tyre with couplant water under it; an "
                             "assumption, not a measured figure")
    parser.add_argument("--safety", type=float, default=3.0)
    parser.add_argument("--gap-mm", type=float, default=0.2)
    parser.add_argument("--plate-mm", type=float, default=None)
    args = parser.parse_args()

    scene = scene_constants("COURSE_MM", "WHEEL_AT", "BEAD_R", "RADIUS",
                            "WHEEL_WIDTH")
    course_mm, wheel_at = scene["COURSE_MM"], scene["WHEEL_AT"]
    bead_r = scene["BEAD_R"]

    print("Mass budget")
    for name, kg in MASS_BUDGET:
        print(f"  {kg:5.2f} kg  {name}")
    print(f"  {args.mass:5.2f} kg  total, with {TETHER_HANG_M:.0f} m of tether "
          f"hanging below it")

    cases = (("dry", args.friction), ("wet, assumed", args.wet_friction))
    gaps = sorted({args.gap_mm, 0.2, 0.6, 1.2})
    sf = f"margin over SF {args.safety:g}"
    worst = {}
    for label, friction in cases:
        print(f"\n{label}: friction {friction}")
        print(f"{'course':>7} {'plate':>7} {'gap':>6} {'need/wheel':>11} "
              f"{'have/wheel':>11} {sf:>17} {'actual':>7}")
        print("-" * 72)
        for index, plate in enumerate(course_mm):
            if args.plate_mm is not None and abs(plate - args.plate_mm) > 1e-6:
                continue
            for gap in gaps:
                got = budget(args.mass, friction, args.safety, gap, plate)
                print(f"{index + 1:>7} {plate:>5.0f} mm {gap:>4.1f} mm "
                      f"{got['per_wheel_need_n']:>9.0f} N "
                      f"{got['per_wheel_available_n']:>9.0f} N "
                      f"{got['margin']:>17.2f} {got['actual_factor']:>7.2f}"
                      f"   {flag(got)}".rstrip())
                if label not in worst or got["margin"] < worst[label][0]["margin"]:
                    worst[label] = (got, index + 1, plate, gap)

    base = budget(args.mass, args.friction, args.safety, 0.2, course_mm[0])
    print(f"\n  sliding needs {base['need_total_sliding_n']:.0f} N across four "
          f"wheels at a friction of {args.friction} and a safety factor of "
          f"{args.safety:g}")
    wet = budget(args.mass, args.wet_friction, args.safety, 0.2, course_mm[0])
    print(f"  wet, at the assumed {args.wet_friction}, it needs "
          f"{wet['need_total_sliding_n']:.0f} N")
    print(f"  peel needs {base['peel_per_upper_wheel_n']:.0f} N on each upper "
          f"wheel, levered by a centre of mass {COM_STANDOFF_M * 1000:.0f} mm "
          f"off the plate across a {base['track_m'] * 1000:.0f} mm track")
    print(f"  the tether adds {base['tether_n']:.0f} N, which is "
          f"{base['tether_n'] / base['weight_n'] * 100:.0f} percent of the "
          f"robot's own weight")

    for label, (got, course, plate, gap) in worst.items():
        print(f"\n  {label}: worst case is course {course}, {plate:.0f} mm plate "
              f"at {gap:.1f} mm of liftoff: margin over SF {got['margin']:.2f}, "
              f"actual factor {got['actual_factor']:.2f}"
              + (f", {flag(got)}" if flag(got) else ""))
        print(f"  {' ' * len(label)}  to hold SF {args.safety:g} there each wheel "
              f"needs a rated pull of {rated_pull_needed(got):.0f} N on bare "
              f"steel, against the {WHEEL_RATED_N:.0f} N assumed")

    # The operating envelope: how much liftoff each plate takes. The table is
    # sampled at three gaps; this is where the line actually falls.
    thin, thick = min(course_mm), max(course_mm)
    print("\n  Operating envelope: the most liftoff under each wheel at which it")
    heads = (f"holds SF, {thin:.0f} mm", f"holds SF, {thick:.0f} mm",
             f"holds at all, {thin:.0f} mm")
    print(f"  {'':<14} {heads[0]:>16} {heads[1]:>17} {heads[2]:>20}")
    for label, friction in cases:
        cells = [largest_gap(args.mass, friction, args.safety, thin, args.safety),
                 largest_gap(args.mass, friction, args.safety, thick, args.safety),
                 largest_gap(args.mass, friction, args.safety, thin, 1.0)]
        text = ["none" if c <= 0.0 else f"{c:.2f} mm" for c in cells]
        print(f"  {label:<14} {text[0]:>16} {text[1]:>17} {text[2]:>20}")
    print(f"  none: not even on bare clean steel ({RATED_AT_GAP_MM:g} mm), with the "
          "rated pull assumed above")

    radius, width = scene["RADIUS"], scene["WHEEL_WIDTH"]
    sagitta_mm = width ** 2 / (8 * radius) * 1000
    print(f"\n  The curvature does not matter. On a {radius:.0f} m radius shell "
          f"a {width * 1000:.0f} mm wide\n  tyre laid across the curve sits in a "
          f"{sagitta_mm:.2f} mm sagitta, which is less than the paint.")
    bead_mm = bead_r * BEAD_HALF_WIDTH_PER_R * 2 * 1000
    print("  The weld bead does matter, and the wheels are placed clear of it:"
          f"\n  they run {max(abs(y) for _, y in wheel_at) * 1000:.0f} mm "
          f"either side of a seam {bead_mm:.0f} mm wide as modelled, so nothing "
          "climbs the bead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
