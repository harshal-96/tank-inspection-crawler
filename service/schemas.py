"""Request and response payloads for the API.

Kept separate from the table models so the wire format can stay stable while the
storage schema moves.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from service.models import Action, DefectType, Station, Verdict


class PartCreate(BaseModel):
    serial: Optional[str] = Field(
        default=None, description="Omit to auto-assign the next P-###### serial"
    )
    station: Station = Station.SIM
    spawn_dy_mm: Optional[float] = None
    spawn_yaw_deg: Optional[float] = None


class MeasurementCreate(BaseModel):
    part_serial: str
    detected_x: float
    detected_y: float
    confidence: float
    n_scans: int = 1
    spread_mm: Optional[float] = None
    latency_ms: Optional[float] = None
    station: Station = Station.SIM
    true_x: Optional[float] = None
    true_y: Optional[float] = None


class InspectionCreate(BaseModel):
    part_serial: str
    image_path: str
    verdict: Verdict
    defect_type: DefectType = DefectType.NONE
    severity: Optional[float] = None
    reasoning: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    injected_defect: Optional[DefectType] = None


class DecisionCreate(BaseModel):
    action: Action
    rationale: str
    part_serial: Optional[str] = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    model: Optional[str] = None
    latency_ms: Optional[float] = None


class ChatRequest(BaseModel):
    message: str
    history_limit: int = 20


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: Optional[str] = None
    latency_ms: Optional[float] = None


class ErrorStats(BaseModel):
    """Correction accuracy over the scored window. Simulation rows only."""

    n: int
    mean_mm: Optional[float] = None
    median_mm: Optional[float] = None
    p95_mm: Optional[float] = None
    max_mm: Optional[float] = None
    sub_mm_rate: Optional[float] = None
    over_3mm: Optional[int] = Field(
        default=None, description="Count of parts above the 3 mm action threshold"
    )


class InspectionStats(BaseModel):
    """Vision model accuracy, scored against injected defects."""

    n_scored: int
    accuracy: Optional[float] = None
    true_pass: int = 0
    false_pass: int = 0
    true_defect: int = 0
    false_defect: int = 0


class DriftSignal(BaseModel):
    """Rolling lateral trend. The thing a single part cannot show you."""

    window: int
    recent_mean_y_mm: Optional[float] = None
    baseline_mean_y_mm: Optional[float] = None
    shift_mm: Optional[float] = None
    trending: bool = False


class CellStats(BaseModel):
    generated_at: datetime
    parts_total: int
    parts_accepted: int
    parts_rework: int
    correction: ErrorStats
    inspection: InspectionStats
    drift: DriftSignal
