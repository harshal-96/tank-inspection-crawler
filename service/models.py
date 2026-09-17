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
