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

import json
import math
import random
import statistics

from cell import geometry as geo
from cell.forming import PressDrift, plan_shift

SCENARIOS = ("press", "fixture", "none")

# The same question with the blind spot closed. A camera whose calibration has
# crept moves every measurement it makes, equally at the bottom of the seam and
# the top, which is the same signature a creeping fixture leaves. Measured on the
# seam alone the two are not merely hard to separate, they are identical, and the
# diagnosis answers "fixture" to both with high confidence. That sends a fitter to
# re-shim a fixture that is fine while the camera mount goes on moving.
#
# Nothing in a cleverer model fixes that, because the information is not in the
# data. It needs an instrument: a reference feature bolted to the machine base,
# not to the fixture, which the camera also sees. If the reference moves, the
# camera moved. If the reference holds still while parts walk, the fixture moved.
SCENARIOS_WITH_CAMERA = ("press", "fixture", "camera", "none")

# How far the reference feature sits from the seam, and the scatter on reading it.
# It is a machined edge rather than a joint, so it reads better than the seam does.
REFERENCE_NOISE_MM = 0.35

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


SYSTEM_PROMPT_WITH_REFERENCE = """You watch a robot welding cell and diagnose what is going wrong.

Each part is measured before welding. The cell tracks the weld seam along its
whole length, so for every part you are given where the seam sits at the BOTTOM of
the joint and where it sits at the TOP, both as an offset in millimetres from
where a perfect part would put it.

You are also given a reference reading: a fixed feature bolted to the machine
base, which the same camera measures on every part. Nothing about the press or the
fixture can move it. Only the camera moving can.

Four things can be happening.

fixture: the part is being located badly. The whole seam moves together, so the
bottom offset and the top offset drift by about the same amount, AND the reference
stays put, because the fixture cannot move a feature bolted to the base. Fix the
fixture or the locators.

camera: the camera mount has crept, so every measurement it makes is shifted. The
seam appears to move exactly as it would for a bad fixture, bottom and top
together, AND the reference drifts with it by the same amount. Nothing is wrong
with the parts at all. Recalibrate the camera.

press: the press brake is going out of adjustment. A bend angle is applied at the
fold and multiplied by the height of the side, so the TOP of the seam walks while
the BOTTOM stays roughly where it was. The tilt, top minus bottom, is what trends.
The reference stays put. Fix the press.

none: nothing is trending. Parts scatter about zero the way any process does.
Say so. Do not invent a fault.

The reference is what separates fixture from camera. On the seam alone those two
are identical, so check it before answering either.

Answer with JSON only:
{"cause": "fixture" | "camera" | "press" | "none",
 "confidence": 0.0 to 1.0,
 "evidence": "one or two sentences naming the numbers that decided it"}"""


# A reply that names something outside the four causes is not a wrong diagnosis,
# it is a reply that did not follow the contract, and the two want counting
# separately. The hard-case study caught a 120B answering "reference" twice, which
# is a channel rather than a cause, and scored it as though the model had picked
# the wrong machine. It had not; it had ignored the answer space.
CAUSE_UNPARSED = "unparsed"
CAUSE_OUT_OF_VOCABULARY = "out_of_vocabulary"


def parse_verdict(raw: str, allowed: tuple[str, ...] = SCENARIOS_WITH_CAMERA) -> dict:
    """Pull a diagnosis out of a reply, and refuse anything outside the answer space.

    Returns the parsed object with "cause" set either to one of the allowed causes
    or to a marker saying why it is not usable, so a caller can report the three
    outcomes apart: right, wrong, and did not answer the question asked.
    """
    text = raw or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {"cause": CAUSE_UNPARSED, "confidence": 0.0, "evidence": text[:200]}
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return {"cause": CAUSE_UNPARSED, "confidence": 0.0, "evidence": text[:200]}
    if not isinstance(data, dict):
        return {"cause": CAUSE_UNPARSED, "confidence": 0.0, "evidence": text[:200]}

    said = str(data.get("cause", "")).strip().lower()
    if said not in allowed:
        data["cause"] = CAUSE_OUT_OF_VOCABULARY
        data["said_instead"] = said
    else:
        data["cause"] = said
    return data


def build_stream(parts: int, scenario: str, seed: int,
                 severity: float = 1.0,
                 reference: bool = False) -> tuple[list[dict], str]:
    """Make a run of parts with a known fault, and the measurements it produces.

    severity scales how far out the machine has gone. At 1.0 the press drift
    moves the tilt by about twice the part-to-part scatter, which is a fault a
    shop would be arguing about rather than one it had already found.

    reference adds the reading of a fixture-independent feature, which is what
    separates a drifting camera from a drifting fixture. Off by default so the
    three-scenario study keeps producing exactly the streams it was scored on.
    """
    rng = random.Random(seed)
    drift = (PressDrift(lean_deg_per_part=0.012 * severity, side="right",
                        starts_at=parts // 4)
             if scenario in ("press", "mixed") else None)
    if scenario not in (*SCENARIOS, "camera", "mixed"):
        raise ValueError(f"unknown scenario {scenario!r}")
    trays = plan_shift(parts, seed=seed, drift=drift)

    rows = []
    for index, tray in enumerate(trays):
        corner = tray.welded_corner()

        # A camera that has crept moves everything it measures, by the same amount
        # at every height and on every feature, including the reference.
        camera_shift = 0.0
        if scenario == "camera" and index >= parts // 4:
            camera_shift = 0.6 * severity * (index - parts // 4)

        # How the part sat in the fixture. Ordinary scatter, plus a creep in one
        # direction if the fixture is the thing going off.
        settle = rng.gauss(0.0, 0.004)
        if scenario in ("fixture", "mixed") and index >= parts // 4:
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
        bottom = ((bottom_dx - bottom_dy) / root_two + settle) * 1000.0 + camera_shift
        top = ((top_dx - top_dy) / root_two + settle) * 1000.0 + camera_shift
        row = {
            "part": index,
            "seam_bottom_mm": round(bottom, 3),
            "seam_top_mm": round(top, 3),
            "tilt_mm": round(top - bottom, 3),
        }
        if reference:
            # Bolted to the machine base, so the press and the fixture cannot move
            # it. Only the camera can.
            row["reference_mm"] = round(
                camera_shift + rng.gauss(0.0, REFERENCE_NOISE_MM), 3)
        rows.append(row)
    return rows, scenario


def format_table(rows: list[dict]) -> str:
    """The measurement stream as the model is shown it.

    The reference column appears only when the run carries one, so a stream built
    without it produces exactly the table the three-scenario study was scored on.
    """
    has_reference = bool(rows) and "reference_mm" in rows[0]
    lines = []
    for r in rows:
        line = (f"part {r['part']:>3}  bottom {r['seam_bottom_mm']:+7.3f}  "
                f"top {r['seam_top_mm']:+7.3f}  tilt {r['tilt_mm']:+7.3f}")
        if has_reference:
            line += f"  reference {r['reference_mm']:+7.3f}"
        lines.append(line)
    return "\n".join(lines)


def user_prompt(rows: list[dict]) -> str:
    return ("Measurements in the order the parts were made, millimetres:\n\n"
            + format_table(rows))


def prompt_for(rows: list[dict]) -> str:
    """The system prompt matching what the rows actually carry.

    Pairing these by hand is how a model ends up told about a reference column
    that is not in its table, or shown one nobody described.
    """
    return (SYSTEM_PROMPT_WITH_REFERENCE
            if rows and "reference_mm" in rows[0] else SYSTEM_PROMPT)


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
    """Every slope the answer turns on, each against its own scatter.

    Two channels when the reference is not instrumented, three when it is. The
    third is the one that separates a drifting camera from a drifting fixture,
    and without it those two are the same measurement.
    """
    tilt, tilt_ratio = slope_ratio([r["tilt_mm"] for r in rows])
    shift, shift_ratio = slope_ratio([r["seam_bottom_mm"] for r in rows])
    out = {"tilt_slope": tilt, "tilt_ratio": tilt_ratio,
           "shift_slope": shift, "shift_ratio": shift_ratio}
    if rows and "reference_mm" in rows[0]:
        ref, ref_ratio = slope_ratio([r["reference_mm"] for r in rows])
        out["reference_slope"] = ref
        out["reference_ratio"] = ref_ratio
    return out


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


# ---------------------------------------------------------------- the hard set
#
# The published evaluation has no marginal case in it. Every run sits at least
# 0.58 away from the decision boundary and most sit one to two away, so a fixed
# threshold cannot fail on it, and every score measured against it is a score on a
# test a ruler passes. These families are built to sit ON the boundary, where a
# ruler and a diagnostician can actually come apart.
#
#   clear          a fault well clear of the threshold. The old kind, kept as a
#                  control so the hard set can be read against something.
#   marginal       a fault that is real and does NOT clear the threshold. Truth is
#                  the injected cause; a ruler says "none".
#   lucky_noise    nothing injected, but scatter pushes a channel over the line.
#                  Truth is "none"; a ruler names a fault.
#   subtle_camera  a camera creeping slowly enough to sit near the line rather
#                  than reading twenty times scatter.
#   mixed          a press AND a fixture going out together. The rule's branches
#                  are exclusive and have nowhere to put this.
#
# Cases are found by rejection: generate, measure the ratio that matters, keep it
# only if it lands in the band. The label is always what was injected, never what
# the ratio says, because the whole point is to find where those two disagree.

HARD_FAMILIES = ("clear", "marginal", "lucky_noise", "subtle_camera", "mixed")

HARD_BANDS = {
    "clear": (3.5, 40.0),
    "marginal": (1.1, 2.4),
    "lucky_noise": (2.0, 4.0),
    "subtle_camera": (1.4, 3.2),
}


def _signal_ratio(scenario: str, d: dict[str, float]) -> float:
    """The channel the injected fault is supposed to move."""
    if scenario == "press":
        return d["tilt_ratio"]
    if scenario == "camera":
        return d.get("reference_ratio", 0.0)
    if scenario == "mixed":
        return min(d["tilt_ratio"], d["shift_ratio"])
    return d["shift_ratio"]


def build_hard_case(family: str, scenario: str, seed: int, parts: int = 30,
                    attempts: int = 400) -> tuple[list[dict], str, dict] | None:
    """One run from a named family, or None if no seed in range landed in the band.

    Returns the rows, the injected truth, and the measured ratios, so a caller can
    report what the case actually looked like rather than asserting it.
    """
    lo, hi = HARD_BANDS.get(family, (0.0, 1e9))
    rng = random.Random(seed)
    for attempt in range(attempts):
        trial_seed = seed * 1000 + attempt
        if is_held_out(trial_seed):
            continue
        if family == "lucky_noise":
            rows, truth = build_stream(parts, "none", trial_seed, 1.0, reference=True)
            d = discriminators(rows)
            loudest = max(d["tilt_ratio"], d["shift_ratio"], d["reference_ratio"])
            if lo <= loudest <= hi:
                return rows, truth, d
            continue

        # A camera creeping is measured against the reference's own scatter, which
        # is far tighter than the seam's, so the severity that puts it near the
        # boundary is an order of magnitude smaller than for the other faults.
        if family == "subtle_camera":
            severity = rng.uniform(0.01, 0.06)
        elif family == "marginal":
            severity = rng.uniform(0.15, 0.55)
        else:
            severity = rng.uniform(1.2, 2.5)
        rows, truth = build_stream(parts, scenario, trial_seed, severity, reference=True)
        d = discriminators(rows)
        ratio = _signal_ratio(scenario, d)
        if not (lo <= ratio <= hi):
            continue
        # A marginal case must be marginal on the signal and quiet elsewhere, or it
        # is just a different fault wearing the wrong label.
        others = [v for k, v in d.items() if k.endswith("_ratio")]
        if family == "marginal" and max(others) > 2.6:
            continue
        if family == "clear" and scenario != "mixed" and sorted(others)[-2] > 2.0:
            continue
        return rows, truth, d
    return None


# The seeds the four published evaluation runs landed on. Training data must not
# come near them, or the result is a model scored on its own homework.
HELD_OUT_SEEDS = (110, 248, 375, 520)
HELD_OUT_MARGIN = 20
TRAINING_SEED_BASE = 20_000


def is_held_out(seed: int) -> bool:
    return any(abs(seed - s) <= HELD_OUT_MARGIN for s in HELD_OUT_SEEDS)
