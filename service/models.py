"""Database schema for the production cell.

One row per part, per measurement, per inspection, per supervisor decision.
Every table that holds a model output also holds a ground-truth column, so
accuracy can be scored offline instead of asserted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import Column, Index
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Station(str, Enum):
    """Where a record came from."""

    SIM = "sim"
    BENCH = "bench"


class PartStatus(str, Enum):
    ARRIVED = "arrived"
    MEASURED = "measured"
    WELDED = "welded"
    INSPECTED = "inspected"
    ACCEPTED = "accepted"
    REWORK = "rework"
    SCRAPPED = "scrapped"


class Verdict(str, Enum):
    PASS = "pass"
    DEFECT = "defect"
    UNCERTAIN = "uncertain"


class DefectType(str, Enum):
    NONE = "none"
    UNDERCUT = "undercut"
    SKIP = "skip"
    POROSITY = "porosity"
    SPATTER = "spatter"
    BURN_THROUGH = "burn_through"


class Action(str, Enum):
    ACCEPT = "accept"
    REWORK = "rework"
    RECALIBRATE = "recalibrate"
    HOLD_LINE = "hold_line"


class Part(SQLModel, table=True):
    """A single part moving through the cell."""

    __tablename__ = "parts"

    id: Optional[int] = Field(default=None, primary_key=True)
    serial: str = Field(index=True, unique=True, description="Human-facing id, e.g. P-000123")
    station: Station = Field(default=Station.SIM, index=True)
    status: PartStatus = Field(default=PartStatus.ARRIVED, index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)

    # Ground-truth arrival pose. Populated in simulation, null on the bench.
    spawn_dy_mm: Optional[float] = None
    spawn_yaw_deg: Optional[float] = None


class Measurement(SQLModel, table=True):
    """Edge perception output: where the detector says the joint actually is."""

    __tablename__ = "measurements"
    __table_args__ = (Index("ix_measurements_part_ts", "part_id", "ts"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    part_id: int = Field(foreign_key="parts.id", index=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    station: Station = Field(default=Station.SIM)

    # Detector output, in robot coordinates (metres).
    detected_x: float
    detected_y: float
    confidence: float
    n_scans: int = 1
    spread_mm: Optional[float] = Field(
        default=None, description="Median-fusion spread across scans, a self-consistency check"
    )
    latency_ms: Optional[float] = None

    # Ground truth. Simulation only, which is what makes the error column trustworthy.
    true_x: Optional[float] = None
    true_y: Optional[float] = None
    error_mm: Optional[float] = Field(default=None, index=True)


class SeamTrack(SQLModel, table=True):
    """Where the seam runs, measured along its whole length.

    A Measurement says where the corner is. This says where the joint GOES, which
    is a different and larger claim: the seam is bowed by the press, so a path
    that is straight by construction walks off it however well the corner was
    located. One row per part, carrying the stations as JSON because the number of
    them is a property of the scan rather than of the schema.
    """

    __tablename__ = "seam_tracks"
    __table_args__ = (Index("ix_seam_tracks_part_ts", "part_id", "ts"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    part_id: int = Field(foreign_key="parts.id", index=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    station: Station = Field(default=Station.SIM)

    stations_measured: int = 0
    stations_asked: int = 0

    # The measured path and the gap at each station, both in the part's own frame.
    # [{"along_mm": .., "x_mm": .., "y_mm": .., "gap_mm": .., "touching": bool}, ..]
    path: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))

    # How far the measured path strays from the straight line a taught program
    # drives. This is the number that a point measurement cannot produce.
    wander_mm: Optional[float] = None
    taught_miss_mm: Optional[float] = Field(
        default=None, description="Worst distance from the taught path to the true seam"
    )

    gap_mm_mean: Optional[float] = None
    gap_mm_max: Optional[float] = None

    # Ground truth, simulation only.
    seam_bow_mm: Optional[float] = Field(
        default=None, description="How far the true seam bows off its own straight line"
    )
    error_mm: Optional[float] = Field(
        default=None, index=True, description="Mean distance from the tracked path to the true seam"
    )
    worst_station_mm: Optional[float] = None

    latency_ms: Optional[float] = None


class WeldSchedule(SQLModel, table=True):
    """What the torch was told to do, station by station, and why.

    The gap is not constant along a seam, so neither are the settings. Stored per
    part rather than per station for the same reason as the path above, and it
    carries the refusals: a cell that never refuses a fit-up is not measuring it.
    """

    __tablename__ = "weld_schedules"

    id: Optional[int] = Field(default=None, primary_key=True)
    part_id: int = Field(foreign_key="parts.id", index=True)
    ts: datetime = Field(default_factory=utcnow, index=True)

    # [{"along_mm": .., "gap_mm": .., "current_a": .., "voltage_v": ..,
    #   "travel_mm_s": .., "heat_j_per_mm": .., "weldable": bool, "reason": ".."}, ..]
    stations: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))

    weldable_stations: int = 0
    refused_stations: int = 0
    flagged_stations: int = 0

    travel_mm_s_min: Optional[float] = None
    travel_mm_s_max: Optional[float] = None
    heat_j_per_mm_max: Optional[float] = None

    # What the heat put into the part is expected to do to the joint while welding.
    predicted_drift_mm: Optional[float] = Field(
        default=None, description="How far the corner walks over the run, from the thermal model"
    )


class ViewDecisionLog(SQLModel, table=True):
    """One decision from the look-and-judge loop, and which model made it.

    Kept because which model answered is part of the answer. The sizes disagree,
    the smallest is the one that is right about this decision, and an operator
    asking why a part took four looks deserves to see the ladder rather than a
    single opaque verdict.
    """

    __tablename__ = "view_decisions"
    __table_args__ = (Index("ix_view_decisions_part_ts", "part_id", "ts"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    part_id: int = Field(foreign_key="parts.id", index=True)
    ts: datetime = Field(default_factory=utcnow, index=True)

    after_looks: int = 1
    accept: bool = True
    next_view_deg: Optional[float] = None
    reason: Optional[str] = None

    model: Optional[str] = Field(default=None, index=True)
    tier: Optional[str] = Field(default=None, index=True,
                                description="nano, lightning, super, ultra or fixed")
    route_why: Optional[str] = None

    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class Inspection(SQLModel, table=True):
    """Post-weld verdict from the cloud vision model, scored against injected truth."""

    __tablename__ = "inspections"

    id: Optional[int] = Field(default=None, primary_key=True)
    part_id: int = Field(foreign_key="parts.id", index=True)
    ts: datetime = Field(default_factory=utcnow, index=True)

    image_path: str
    verdict: Verdict = Field(index=True)
    defect_type: DefectType = Field(default=DefectType.NONE)
    severity: Optional[float] = Field(default=None, description="0 to 1, model's own estimate")
    reasoning: Optional[str] = None

    model: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Ground truth from the defect injector. Null for bench captures.
    injected_defect: Optional[DefectType] = None
    correct: Optional[bool] = Field(default=None, index=True)


class Decision(SQLModel, table=True):
    """Supervisor agent output. May target one part or the line as a whole."""

    __tablename__ = "decisions"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    part_id: Optional[int] = Field(default=None, foreign_key="parts.id", index=True)

    action: Action = Field(index=True)
    rationale: str
    evidence: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    model: Optional[str] = None
    latency_ms: Optional[float] = None
    acknowledged: bool = Field(default=False)


class AgentMessage(SQLModel, table=True):
    """Operator chat transcript, including the agent's tool calls."""

    __tablename__ = "agent_messages"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    role: str = Field(description="user, assistant, tool or system")
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    tool_call_id: Optional[str] = None
    model: Optional[str] = None


class ShiftReport(SQLModel, table=True):
    """Generated end-of-shift summary."""

    __tablename__ = "shift_reports"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    period_start: datetime
    period_end: datetime
    content_md: str
    model: Optional[str] = None


class CorrectiveBrief(SQLModel, table=True):
    """A cited answer to "what do I do about this", raised by a measurement.

    Kept because the citation is the point. A verdict an operator cannot check is
    a verdict nobody on a shop floor is allowed to act on, so the sources the
    brief was written from are stored beside it rather than being thrown away
    once the text was generated.
    """

    __tablename__ = "corrective_briefs"

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utcnow, index=True)
    part_id: Optional[int] = Field(default=None, foreign_key="parts.id", index=True)

    topic: str = Field(index=True,
                       description="What raised it, e.g. the joint opening too far to weld")
    trigger: str = Field(default="", description="gap, drift, defect: which measurement asked")
    query: str = Field(description="What was searched, built from the measurement")

    headline: str = ""
    what_the_cell_measured: str = ""
    what_the_sources_say: str = ""
    do_now: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    confidence: str = "low"
    caveat: str = ""

    sources: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    documents_read: int = 0
    model: Optional[str] = None
    search_ms: Optional[float] = None
    total_ms: Optional[float] = None
    error: Optional[str] = None
