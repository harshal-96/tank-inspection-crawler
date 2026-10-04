"""What holds the crawler on the wall, and where that stops being true.

    python tank/adhesion.py
    python tank/adhesion.py --mass 8.0 --friction 0.25 --gap-mm 1.0

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

And two things quietly destroy all of it.

  liftoff   holding force falls roughly with the square of the gap. Paint,
            mill scale and rust are gap. A millimetre of corrosion product can
            halve the grip, and the places worth inspecting are exactly the
            places that have it.
  thin plate a magnetic circuit needs steel to carry its flux. Below roughly
            six to eight millimetres the plate saturates and the magnet cannot
            develop its rated pull however good it is. This tank steps from
            12 mm at the bottom course to 6 mm at the top, so the grip is
            weakest at the top, which is also furthest to fall.
"""

from __future__ import annotations

import argparse
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
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
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

# What a decent 78 mm magnetic wheel develops on thick, clean, bare steel.
# Treated as a quoted figure to be derated, not as a result.
WHEEL_RATED_N = 220.0

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

    available = WHEEL_RATED_N * plate_derate(plate_mm) * gap_derate(gap_mm)
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
        "margin": available / per_wheel_need if per_wheel_need else math.inf,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float,
                        default=sum(m for _, m in MASS_BUDGET))
    parser.add_argument("--friction", type=float, default=0.35,
                        help="rubber tyred magnetic wheel on steel, dry")
    parser.add_argument("--safety", type=float, default=3.0)
    parser.add_argument("--gap-mm", type=float, default=0.2)
    parser.add_argument("--plate-mm", type=float, default=None)
    args = parser.parse_args()

    scene = scene_constants("COURSE_MM", "WHEEL_AT", "BEAD_R")
    course_mm, wheel_at = scene["COURSE_MM"], scene["WHEEL_AT"]
    bead_r = scene["BEAD_R"]

    print("Mass budget")
    for name, kg in MASS_BUDGET:
        print(f"  {kg:5.2f} kg  {name}")
    print(f"  {args.mass:5.2f} kg  total, with {TETHER_HANG_M:.0f} m of tether "
          f"hanging below it\n")

    print(f"{'course':>7} {'plate':>7} {'gap':>6} {'need/wheel':>11} "
          f"{'have/wheel':>11} {'margin':>7}")
    print("-" * 56)
    worst = None
    for index, plate in enumerate(course_mm):
        if args.plate_mm is not None and abs(plate - args.plate_mm) > 1e-6:
            continue
        for gap in sorted({args.gap_mm, 0.2, 0.6, 1.2}):
            got = budget(args.mass, args.friction, args.safety, gap, plate)
            flag = "" if got["margin"] >= 1.0 else "   FAILS"
            print(f"{index + 1:>7} {plate:>5.0f} mm {gap:>4.1f} mm "
                  f"{got['per_wheel_need_n']:>9.0f} N {got['per_wheel_available_n']:>9.0f} N "
                  f"{got['margin']:>6.2f}{flag}")
            if worst is None or got["margin"] < worst[0]:
                worst = (got["margin"], index + 1, plate, gap)

    base = budget(args.mass, args.friction, args.safety, 0.2, course_mm[0])
    print(f"\n  sliding needs {base['need_total_sliding_n']:.0f} N across four "
          f"wheels at a friction of {args.friction} and a safety factor of "
          f"{args.safety:.0f}")
    print(f"  peel needs {base['peel_per_upper_wheel_n']:.0f} N on each upper "
          f"wheel, levered by a centre of mass {COM_STANDOFF_M * 1000:.0f} mm "
          f"off the plate across a {base['track_m'] * 1000:.0f} mm track")
    print(f"  the tether adds {base['tether_n']:.0f} N, which is "
          f"{base['tether_n'] / base['weight_n'] * 100:.0f} percent of the "
          f"robot's own weight")
    if worst:
        print(f"\n  worst case is course {worst[1]}, {worst[2]:.0f} mm plate at "
              f"{worst[3]:.1f} mm of liftoff: margin {worst[0]:.2f}")
    print("\n  The curvature does not matter. On a 10 m radius shell a 52 mm "
          "wheel\n  sits in a 0.03 mm sagitta, which is less than the paint.")
    print("  The weld bead does matter, and the wheels are placed clear of it: "
          f"\n  they run {max(abs(y) for _, y in wheel_at) * 1000:.0f} mm "
          f"either side of a seam {bead_r * 2000:.0f} mm wide, so nothing "
          "climbs the bead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
