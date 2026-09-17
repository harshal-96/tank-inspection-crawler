"""The measurement stream a run of parts produces, and what it means.

This is the single source for the drift question. The evaluation, the training
data and the cell's own diagnosis all import from here, so a model cannot be
trained on one description of the problem and scored against another. That is not
a hypothetical worry: the two halves were written weeks apart, and the only reason
they can be compared is that they read the same generator.

The physics of the discrimination, which is the whole reason the question is
answerable at all:

  a fixture going off    moves the whole seam together. The bottom of the joint
                         and the top of it shift by the same amount, because the
                         part is sitting in the wrong place and is otherwise fine.

  a press going off      moves the top of the joint and leaves the bottom where
                         it was, because a bend angle is applied at a fold and
                         multiplied by height. The seam tilts rather than shifts.

A cell that only locates a point cannot tell those apart. A cell that tracks the
whole seam gets the discriminator for free.
"""

from __future__ import annotations

import math
import random
import statistics

from cell import geometry as geo
from cell.forming import PressDrift, plan_shift

SCENARIOS = ("press", "fixture", "none")

SYSTEM_PROMPT = """You watch a robot welding cell and diagnose what is going wrong.

Each part is measured before welding. The cell tracks the weld seam along its
whole length, so for every part you are given where the seam sits at the BOTTOM of
the joint and where it sits at the TOP, both as an offset in millimetres from
where a perfect part would put it.

Three things can be happening.

fixture: the part is being located badly. The whole seam moves together, so the
bottom offset and the top offset drift by about the same amount. Fix the fixture
or the locators.

press: the press brake is going out of adjustment. A bend angle is applied at the
fold and multiplied by the height of the side, so the TOP of the seam walks while
the BOTTOM stays roughly where it was. The tilt, top minus bottom, is what trends.
Fix the press.

none: nothing is trending. Parts scatter about zero the way any process does.
Say so. Do not invent a fault.

Answer with JSON only:
{"cause": "fixture" | "press" | "none",
 "confidence": 0.0 to 1.0,
 "evidence": "one or two sentences naming the numbers that decided it"}"""


def build_stream(parts: int, scenario: str, seed: int,
                 severity: float = 1.0) -> tuple[list[dict], str]:
    """Make a run of parts with a known fault, and the measurements it produces.

    severity scales how far out the machine has gone. At 1.0 the press drift
    moves the tilt by about twice the part-to-part scatter, which is a fault a
    shop would be arguing about rather than one it had already found.
    """
    rng = random.Random(seed)
    drift = (PressDrift(lean_deg_per_part=0.012 * severity, side="right",
                        starts_at=parts // 4)
             if scenario == "press" else None)
    trays = plan_shift(parts, seed=seed, drift=drift)

    rows = []
    for index, tray in enumerate(trays):
        corner = tray.welded_corner()

        # How the part sat in the fixture. Ordinary scatter, plus a creep in one
        # direction if the fixture is the thing going off.
        settle = rng.gauss(0.0, 0.004)
        if scenario == "fixture" and index >= parts // 4:
            settle += 0.0006 * severity * (index - parts // 4)

        # Offsets of the seam from nominal, at the bottom of the joint and the top,
        # measured along the direction the corner actually moves: outward on the
        # diagonal between the two sides.
        #
        # Taking only the y component, as this did at first, hides a fault on the
        # side that moves the seam in x, and the press scenario came out with
        # exactly the same numbers as the control. A fault the measurement cannot
        # see is not a fault the model can be scored on.
        root_two = math.sqrt(2.0)
        bottom_dx, bottom_dy = corner.seam_offset_m(0.0, geo.PART_D)
        top_dx, top_dy = corner.seam_offset_m(geo.PART_D, geo.PART_D)
        bottom = ((bottom_dx - bottom_dy) / root_two + settle) * 1000.0
        top = ((top_dx - top_dy) / root_two + settle) * 1000.0
        rows.append({
            "part": index,
            "seam_bottom_mm": round(bottom, 3),
            "seam_top_mm": round(top, 3),
            "tilt_mm": round(top - bottom, 3),
        })
    return rows, scenario


def format_table(rows: list[dict]) -> str:
    """The measurement stream as the model is shown it."""
    return "\n".join(
        f"part {r['part']:>3}  bottom {r['seam_bottom_mm']:+7.3f}  "
        f"top {r['seam_top_mm']:+7.3f}  tilt {r['tilt_mm']:+7.3f}"
        for r in rows
    )


def user_prompt(rows: list[dict]) -> str:
    return ("Measurements in the order the parts were made, millimetres:\n\n"
            + format_table(rows))


def trend(values: list[float]) -> float:
    """Least squares slope, as a sanity check the model has something to find."""
    n = len(values)
    xs = list(range(n))
    mx, my = statistics.mean(xs), statistics.mean(values)
    denominator = sum((x - mx) ** 2 for x in xs)
    return 0.0 if denominator == 0 else sum(
        (x - mx) * (y - my) for x, y in zip(xs, values, strict=True)) / denominator


def slopes(rows: list[dict]) -> tuple[float, float]:
    """The two slopes the answer turns on: how the seam tilts, and how it shifts."""
    return (trend([r["tilt_mm"] for r in rows]),
            trend([r["seam_bottom_mm"] for r in rows]))


def slope_ratio(values: list[float]) -> tuple[float, float]:
    """The slope, and how many of its own standard errors it stands away from flat.

    A slope on its own is not evidence. Part-to-part scatter here is several
    millimetres, so a least squares line through thirty noisy parts picks up a
    visible slope perfectly often with nothing wrong at all. What separates a
    fault from a run of luck is the slope measured against the scatter it was
    drawn from, and that ratio is what the labels are written from.
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0
    slope = trend(values)
    xs = list(range(n))
    mx, my = statistics.mean(xs), statistics.mean(values)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return slope, 0.0
    intercept = my - slope * mx
    residuals = [v - (slope * x + intercept) for x, v in zip(xs, values, strict=True)]
    dof = n - 2
    sigma = math.sqrt(sum(r * r for r in residuals) / dof) if dof > 0 else 0.0
    stderr = sigma / math.sqrt(sxx)
    if stderr > 0:
        return slope, abs(slope) / stderr
    # No scatter at all. A flat line is flat; a perfect ramp is as certain as a
    # trend can be, and returning zero for it would call the clearest possible
    # signal no signal. Real measurements never land here, but a helper whose job
    # is to say whether a trend is real should not answer backwards when they do.
    return slope, (0.0 if slope == 0 else float("inf"))


def discriminators(rows: list[dict]) -> dict[str, float]:
    """Both slopes and both ratios, which is everything the answer turns on."""
    tilt, tilt_ratio = slope_ratio([r["tilt_mm"] for r in rows])
    shift, shift_ratio = slope_ratio([r["seam_bottom_mm"] for r in rows])
    return {"tilt_slope": tilt, "tilt_ratio": tilt_ratio,
            "shift_slope": shift, "shift_ratio": shift_ratio}


# How far a slope has to stand out of the scatter before a run is worth training
# on as an example of that fault. Below this the fault was injected but did not
# show, and labelling it teaches the model to answer without evidence.
VISIBLE_RATIO = 2.0


def is_quiet(rows: list[dict]) -> bool:
    """Does this run really have nothing in it?

    A control drawn at random is not automatically a control. Ordinary scatter
    produces trends by chance, and the first version of the evaluation scored a
    model against a run labelled "nothing wrong" that had a visible tilt in it.
    The model called that a press fault, which on the evidence in front of it was
    a reasonable answer to a question that had been asked wrongly.
    """
    tilt, shift = slopes(rows)
    return abs(tilt) < 0.008 and abs(shift) < 0.030


def quiet_seed(parts: int, start: int) -> int:
    """Find a seed whose no-fault run really is flat."""
    for offset in range(400):
        seed = start + offset
        rows, _ = build_stream(parts, "none", seed)
        if is_quiet(rows):
            return seed
    raise SystemExit("no quiet seed found; widen the search")


# The seeds the four published evaluation runs landed on. Training data must not
# come near them, or the result is a model scored on its own homework.
HELD_OUT_SEEDS = (110, 248, 375, 520)
HELD_OUT_MARGIN = 20
TRAINING_SEED_BASE = 20_000


def is_held_out(seed: int) -> bool:
    return any(abs(seed - s) <= HELD_OUT_MARGIN for s in HELD_OUT_SEEDS)
