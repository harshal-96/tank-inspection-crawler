"""Where to put the thickness probe next, decided by an NVIDIA Nemotron model.

The crawler carries two instruments and they disagree about what matters. The
camera sees the surface: rust, a groove at a weld toe, a pore in a bead. The
ultrasonic probe sees through the plate and reports what is left of it. The
run in tank/data/survey is built on that premise on purpose. It carries six
patches of wall loss: one with rust showing on the surface, which measures inside
limits, and five with nothing on the surface, which hold all 28 of the 168
probe points that fail the screening limit. 20 of those 28 sit under frames
the inspector passes. tank/score_probes.py prints these counts from the
manifest.

So probing on the seam, or beside the rust the camera reports, finds 2 of the
28 (score_probes.py --baselines). A planner has to cover the band the arm can
reach, notice a thin reading when it gets one, and go back to it. This module
asks a text model to make that choice, and sweep_rule below makes it with
arithmetic. The model was never ahead in a recorded run: it found 9 to 12 of
28, and the sweep 16 at its shipped tie-break order, a lucky one (median 12
over all 5040 orders, score_probes.py --tiebreaks). No run differs from the
sweep significantly (tank/significance.py). So the rule drives the arm because
it is no worse on this evidence, deterministic and free, and this planner is
kept as the measured alternative.

The arm offers seven places across the seam at every station and there is time
for two, so this is 2 of 7, twenty four times, with the answers written before
it started.

Nothing here is allowed to see the ground truth. The planner is given what the
crawler knows: what the inspector said about the picture, which offsets exist,
which have never been measured, and the last six readings on this run. The
sweep reads the whole run's history, so the two are not matched on
information; score_probes.py --baselines has a "sweep, model's view" arm that
holds the rule to the model's six readings. Withholding that history does not
hurt the rule: over all tie-break orders it finds a median of 13 shown only the
model's view against 12 with the whole history (score_probes.py --tiebreaks),
so the state the sweep keeps is not what separates it from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from brain import nebius

NL = chr(10)

# How many of the seven the arm has time for at one station. An assumption,
# not a measured cycle time: a spot probe that is placed, coupled and read at
# each place. A probe that reads continuously as it rolls would take the
# choice away altogether.
DEFAULT_BUDGET = 2

# A reading at or below this share of nominal is worth going back to, even
# though it is not yet a rejection. The screening limit itself is 0.60 of
# nominal, a threshold chosen for this benchmark rather than an API 653
# retirement thickness, and a planner that only reacts at 0.60 reacts one
# station too late. This is the rule's own setting: the model's prompt gives
# the reject level and never this number.
WATCH_FRACTION = 0.80

SYSTEM_PROMPT = """You are planning ultrasonic thickness measurements for a \
robot crawling a weld seam inside a steel storage tank.

At each station the arm can reach seven places across the seam, given as \
offsets in metres. A negative offset is below the seam, a positive offset is \
above it, and 0.0 is on the seam itself. There is only time for a few of them, \
so you choose.

What you are looking for is wall loss: metal eaten away from the far side, \
where nothing shows on the surface the camera can see. A frame the inspector \
calls sound tells you the surface is clean. It tells you nothing about the \
thickness behind it, and the whole reason this robot carries a probe is that \
those two are different questions.

How to choose:
- Cover the band. An offset never measured on this run is worth more than one \
measured two stations ago, because loss you have not looked for is the loss \
that gets missed.
- Chase a thin reading. If a recent measurement came back low, measure at or \
beside that offset again: wall loss comes in patches, so a thin point means \
the next station probably has one too.
- Corrosion the camera can see is worth a look, but it is not where the \
dangerous loss usually is, and it must never cost you coverage of the rest of \
the band.

Reply with JSON only, no prose and no code fence:
{"offsets": [the offsets you choose, exactly as given in the list],
 "why": "one short sentence"}

Answer with the JSON object and nothing else. Do not write your reasoning out as prose first: the one sentence in "why" is all the room there is for it."""  # noqa: E501  the recorded runs used this exact text

# Which NVIDIA model plans the probing. Nano and Super both answered cleanly.
# Nemotron 3.5 Lightning, the project's default for text, wrote its reasoning
# out in prose and did not reach the JSON in the runs tried, but plan_probes
# makes its call with default reasoning: unlike the cell's other Nemotron
# calls it passes no reasoning_effort. So that is a note about this envelope,
# not evidence that Lightning cannot plan, and the recorded Nano runs were made
# in the same envelope. The envelope changes per model; the question does not.
DEFAULT_PLANNER_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

# What plan_probes writes in `why` when the reply could not be used and the
# sweep chose instead. The recorded runs made before fell_back was stored carry
# only these words, so score_probes.fell_back_stations reads them back from
# there; change them and the old runs can no longer be audited.
UNREADABLE_WHY = "unreadable reply, swept instead"
EMPTY_WHY = "no usable offsets in reply, swept instead"
FALLBACK_WHY = (UNREADABLE_WHY, EMPTY_WHY)


def planner_model() -> str:
    """The planner's model, overridable without touching the code."""
    import os
    return os.getenv("PROBE_PLANNER_MODEL", "").strip() or DEFAULT_PLANNER_MODEL


@dataclass
class ProbePlan:
    offsets: list[float]
    why: str = ""
    model: Optional[str] = None
    latency_ms: float = 0.0
    fell_back: bool = False
    raw: str = ""
    # The reply named fewer offsets than the budget and the sweep supplied the
    # rest. Recorded per station by score_probes.py from now on; the runs
    # recorded before this field existed cannot show it.
    padded: bool = False


@dataclass
class Reading:
    station: int
    offset_m: float
    thickness_mm: float
    nominal_mm: float

    @property
    def fraction(self) -> float:
        return self.thickness_mm / self.nominal_mm


@dataclass
class Memory:
    """What the crawler has measured so far on this run."""

    readings: list[Reading] = field(default_factory=list)

    def last_seen(self, offset: float) -> Optional[Reading]:
        for r in reversed(self.readings):
            if abs(r.offset_m - offset) < 1e-6:
                return r
        return None

    def stations_since(self, offset: float, station: int) -> Optional[int]:
        seen = self.last_seen(offset)
        return None if seen is None else station - seen.station

    def thin_offsets(self, within: int, station: int) -> list[float]:
        """Offsets that read low recently enough to be worth another look."""
        out = []
        for r in reversed(self.readings):
            if station - r.station > within:
                break
            if r.fraction <= WATCH_FRACTION and r.offset_m not in out:
                out.append(r.offset_m)
        return out


def sweep_rule(offsets: list[float], station: int, memory: Memory,
               budget: int) -> list[float]:
    """The arithmetic the planner is measured against, and falls back to.

    Chase anything that read thin in the last two stations, then spend what is
    left on whichever offsets have gone longest without a measurement. It is
    not a strawman: covering the band in rotation is what a careful person with
    a clipboard would do, and a model that cannot beat it is not earning the
    call it costs.
    """
    picked: list[float] = []
    for offset in memory.thin_offsets(within=2, station=station):
        if offset in offsets and offset not in picked and len(picked) < budget:
            picked.append(offset)

    def staleness(offset: float) -> tuple[int, float]:
        since = memory.stations_since(offset, station)
        return (-(10 ** 6) if since is None else -since, abs(offset))

    for offset in sorted(offsets, key=staleness):
        if len(picked) >= budget:
            break
        if offset not in picked:
            picked.append(offset)
    return picked[:budget]


def _coerce(data: dict[str, Any], offsets: list[float],
            budget: int) -> list[float]:
    """Keep only offsets that exist, without inventing the rest."""
    wanted = data.get("offsets")
    if isinstance(wanted, (int, float, str)):
        wanted = [wanted]
    picked: list[float] = []
    for item in wanted or []:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        match = min(offsets, key=lambda o: abs(o - value))
        if abs(match - value) < 0.02 and match not in picked:
            picked.append(match)
    return picked[:budget]


def plan_probes(station: int, offsets: list[float], nominal_mm: float,
                limit_fraction: float, inspector_said: str, memory: Memory,
                budget: int = DEFAULT_BUDGET, model: Optional[str] = None,
                api: Optional[Any] = None) -> ProbePlan:
    """One station in, the offsets to measure out.

    An unreadable reply falls back to the sweep rule rather than to a default
    offset, for the same reason the rest of this project does it: a planner
    that quietly measures the middle when it could not think is a planner that
    reports a clean run it never made.
    """
    import time

    model = model or planner_model()
    recent = memory.readings[-6:]
    history = "\n".join(
        f"  station {r.station} at {r.offset_m:+.2f} read {r.thickness_mm:.2f} mm "
        f"of {r.nominal_mm:.1f} ({r.fraction * 100:.0f} percent left)" for r in recent
    ) or "  nothing measured yet"
    never = [o for o in offsets if memory.last_seen(o) is None]

    # Kept short and flat. Given the same facts spread over a longer
    # message the model spent its whole budget re-reading them and
    # arguing with itself about which list was which.
    user = (
        f"Station {station}.{NL}"
        f"Picture: {inspector_said}.{NL}"
        f"Offsets available: {offsets}{NL}"
        f"Not yet measured on this run: {never or 'none'}{NL}"
        f"Recent readings:{NL}{history}{NL}"
        f"Nominal wall {nominal_mm:.1f} mm, reject at or below "
        f"{nominal_mm * limit_fraction:.1f} mm.{NL}{NL}"
        f"Choose exactly {budget} offsets from the available list."
    )

    started = time.perf_counter()
    reply = nebius.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user}],
        model=model, max_tokens=900, parses=nebius.extract_json, api=api)
    elapsed = (time.perf_counter() - started) * 1000

    try:
        data = nebius.extract_json(reply.raw)
    except Exception:  # noqa: BLE001  any unreadable reply
        return ProbePlan(sweep_rule(offsets, station, memory, budget),
                         UNREADABLE_WHY, model, elapsed,
                         fell_back=True, raw=reply.raw)

    picked = _coerce(data, offsets, budget)
    if not picked:
        return ProbePlan(sweep_rule(offsets, station, memory, budget),
                         EMPTY_WHY, model,
                         elapsed, fell_back=True, raw=reply.raw)
    # Short of budget is not a failure of judgement, but the arm has the time,
    # so the rest goes to the stalest offsets rather than being left unused.
    # Marked, so a score can tell the model's picks from the rule's.
    padded = len(picked) < budget
    if padded:
        for offset in sweep_rule(offsets, station, memory, budget):
            if offset not in picked and len(picked) < budget:
                picked.append(offset)
    return ProbePlan(picked, str(data.get("why") or "").strip()[:160],
                     model, elapsed, raw=reply.raw, padded=padded)


def plan_with_floor(station: int, offsets: list[float], nominal_mm: float,
                    limit_fraction: float, inspector_said: str, memory: Memory,
                    budget: int = DEFAULT_BUDGET, model: Optional[str] = None,
                    api: Optional[Any] = None) -> ProbePlan:
    """The rule claims one measurement, the model spends the rest.

    Measured only on the retired four point scene (tank/data/survey/
    archive_four_point). There the model alone found 4, 3 and 4 of the 4
    rejectable points against the sweep's 2, and with this floor it found 4,
    4, 2, 3 and 4 over five runs. It has not been run on the 28 point scene,
    where the model alone found fewer than the sweep in every recorded run, so
    nothing here says how the floor does there.

    The idea: the useful property of the sweep is not that it is clever, it is
    that it is the same every time. So the sweep's first pick is reserved. It
    is the offset that has gone longest without a reading, or one that came
    back thin in the last two stations, which is the coverage the model is
    most likely to drop when it becomes interested in something. The model
    fills what is left. The plan always includes the sweep's first pick at
    every station; that is all it guarantees; the sweep's second pick can
    still be lost, so this is not a promise to match the sweep's score.
    """
    floor = sweep_rule(offsets, station, memory, 1)
    if budget <= 1:
        return ProbePlan(floor, "budget of one, the rule takes it", model)

    plan = plan_probes(station, offsets, nominal_mm, limit_fraction,
                       inspector_said, memory, budget=budget, model=model, api=api)
    picked = list(floor)
    for offset in plan.offsets:
        if len(picked) >= budget:
            break
        if offset not in picked:
            picked.append(offset)
    # The model may have agreed with the rule, in which case there is room
    # left; the sweep fills it rather than leaving the arm idle.
    if len(picked) < budget:
        for offset in sweep_rule(offsets, station, memory, budget):
            if offset not in picked and len(picked) < budget:
                picked.append(offset)
    return ProbePlan(picked, f"rule floor {floor[0]:+.2f}; {plan.why}"[:160],
                     plan.model, plan.latency_ms, plan.fell_back, plan.raw,
                     padded=plan.padded)
