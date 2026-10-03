"""Where to put the thickness probe next, decided by an NVIDIA Nemotron model.

The crawler carries two instruments and they disagree about what matters. The
camera sees the surface: rust, a groove at a weld toe, a pore in a bead. The
ultrasonic probe sees through the plate and reports what is left of it. Neither
is enough on its own, and the run in tank/data/survey says why in one line: of
the four probe points on it that fail the acceptance limit, every one is on a
patch with nothing on the surface at all, under frames the inspector reads as
clean, while the patch that is plainly rusty measures inside limits.

So a planner that probes where the camera saw something finds nothing, and a
planner that always probes on the seam finds nothing either. It has to cover
the band the arm can reach, notice a thin reading when it gets one, and go back
to it. That is a decision about where to spend a scarce measurement, which is
the kind of thing a text model can do and a fixed pattern cannot.

The arm offers seven places across the seam at every station and there is time
for two, so this is 2 of 7, twenty four times, with the answers written before
it started.

Nothing here is allowed to see the ground truth. The planner is given what the
crawler knows: what the inspector said about the picture, which offsets exist,
and the readings already taken on this run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from brain import nebius

NL = chr(10)

# How many of the seven the arm has time for at one station.
DEFAULT_BUDGET = 2

# A reading at or below this share of nominal is worth going back to, even
# though it is not yet a rejection. The acceptance limit itself is 0.60, and a
# planner that only reacts at 0.60 reacts one station too late.
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

Answer with the JSON object and nothing else. Do not write your reasoning out as prose first: the one sentence in "why" is all the room there is for it."""

# Which NVIDIA model plans the probing. Nemotron 3.5 Lightning, the project's
# default for text, will not do this job: given a station it writes its
# reasoning out in prose and never reaches the JSON, with three thousand tokens
# of room as readily as with three hundred. Nano and Super both answer cleanly.
# The envelope changes per model; the question does not.
DEFAULT_PLANNER_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


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
                         "unreadable reply, swept instead", model, elapsed,
                         fell_back=True, raw=reply.raw)

    picked = _coerce(data, offsets, budget)
    if not picked:
        return ProbePlan(sweep_rule(offsets, station, memory, budget),
                         "no usable offsets in reply, swept instead", model,
                         elapsed, fell_back=True, raw=reply.raw)
    # Short of budget is not a failure of judgement, but the arm has the time,
    # so the rest goes to the stalest offsets rather than being left unused.
    if len(picked) < budget:
        for offset in sweep_rule(offsets, station, memory, budget):
            if offset not in picked and len(picked) < budget:
                picked.append(offset)
    return ProbePlan(picked, str(data.get("why") or "").strip()[:160],
                     model, elapsed, raw=reply.raw)
