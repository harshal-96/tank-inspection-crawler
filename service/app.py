"""FastAPI service for the production cell.

Single source of truth. The simulation writes to it, the dashboard reads from it,
and the supervisor agent's tools query it. Nothing keeps its own copy of the state.
"""

from __future__ import annotations

import statistics
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, func, select

from service.db import get_session, init_db
from service.models import (
    Decision,
    Inspection,
    Measurement,
    Part,
    PartStatus,
    Station,
    Verdict,
    utcnow,
)
from service.schemas import (
    CellStats,
    DecisionCreate,
    DriftSignal,
    ErrorStats,
    InspectionCreate,
    InspectionStats,
    MeasurementCreate,
    PartCreate,
)

load_dotenv()

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Self-Correcting Production Cell",
    description="Measurement, inspection and supervision for a robotic welding cell.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- helpers


def _part_by_serial(session: Session, serial: str) -> Part:
    part = session.exec(select(Part).where(Part.serial == serial)).first()
    if part is None:
        raise HTTPException(status_code=404, detail=f"No part with serial {serial!r}")
    return part


def _next_serial(session: Session) -> str:
    count = session.exec(select(func.count()).select_from(Part)).one()
    return f"P-{count + 1:06d}"


def _error_mm(
    detected_x: float, detected_y: float, true_x: Optional[float], true_y: Optional[float]
) -> Optional[float]:
    """Planar distance between the detected joint and the real one, in millimetres."""
    if true_x is None or true_y is None:
        return None
    return 1000.0 * ((detected_x - true_x) ** 2 + (detected_y - true_y) ** 2) ** 0.5


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Stable for the small n a shift produces."""
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[idx]


# --------------------------------------------------------------------------- health


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


# --------------------------------------------------------------------------- parts


@app.post("/api/parts", response_model=Part, status_code=201)
def create_part(payload: PartCreate, session: Session = Depends(get_session)) -> Part:
    serial = payload.serial or _next_serial(session)
    if session.exec(select(Part).where(Part.serial == serial)).first():
        raise HTTPException(status_code=409, detail=f"Serial {serial!r} already exists")

    part = Part(
        serial=serial,
        station=payload.station,
        spawn_dy_mm=payload.spawn_dy_mm,
        spawn_yaw_deg=payload.spawn_yaw_deg,
    )
    session.add(part)
    session.commit()
    session.refresh(part)
    return part


@app.get("/api/parts", response_model=list[Part])
def list_parts(
    limit: int = 50,
    station: Optional[Station] = None,
    session: Session = Depends(get_session),
) -> list[Part]:
    query = select(Part).order_by(Part.id.desc()).limit(min(limit, 500))
    if station is not None:
        query = query.where(Part.station == station)
    return list(session.exec(query))


@app.get("/api/parts/{serial}", response_model=Part)
def get_part(serial: str, session: Session = Depends(get_session)) -> Part:
    return _part_by_serial(session, serial)


# --------------------------------------------------------------------- measurements


@app.post("/api/measurements", response_model=Measurement, status_code=201)
def create_measurement(
    payload: MeasurementCreate, session: Session = Depends(get_session)
) -> Measurement:
    part = _part_by_serial(session, payload.part_serial)

    measurement = Measurement(
        part_id=part.id,
        station=payload.station,
        detected_x=payload.detected_x,
        detected_y=payload.detected_y,
        confidence=payload.confidence,
        n_scans=payload.n_scans,
        spread_mm=payload.spread_mm,
        latency_ms=payload.latency_ms,
        true_x=payload.true_x,
        true_y=payload.true_y,
        error_mm=_error_mm(
            payload.detected_x, payload.detected_y, payload.true_x, payload.true_y
        ),
    )
    part.status = PartStatus.MEASURED
    session.add_all([measurement, part])
    session.commit()
    session.refresh(measurement)
    return measurement


@app.get("/api/measurements", response_model=list[Measurement])
def list_measurements(
    limit: int = 50, session: Session = Depends(get_session)
) -> list[Measurement]:
    query = select(Measurement).order_by(Measurement.id.desc()).limit(min(limit, 500))
    return list(session.exec(query))


# ---------------------------------------------------------------------- inspections


@app.post("/api/inspections", response_model=Inspection, status_code=201)
def create_inspection(
    payload: InspectionCreate, session: Session = Depends(get_session)
) -> Inspection:
    part = _part_by_serial(session, payload.part_serial)

    correct: Optional[bool] = None
    if payload.injected_defect is not None:
        clean = payload.injected_defect.value == "none"
        called_pass = payload.verdict == Verdict.PASS
        correct = clean == called_pass

    inspection = Inspection(
        part_id=part.id,
        image_path=payload.image_path,
        verdict=payload.verdict,
        defect_type=payload.defect_type,
        severity=payload.severity,
        reasoning=payload.reasoning,
        model=payload.model,
        latency_ms=payload.latency_ms,
        prompt_tokens=payload.prompt_tokens,
        completion_tokens=payload.completion_tokens,
        injected_defect=payload.injected_defect,
        correct=correct,
    )
    part.status = PartStatus.INSPECTED
    session.add_all([inspection, part])
    session.commit()
    session.refresh(inspection)
    return inspection


@app.get("/api/inspections", response_model=list[Inspection])
def list_inspections(
    limit: int = 50, session: Session = Depends(get_session)
) -> list[Inspection]:
    query = select(Inspection).order_by(Inspection.id.desc()).limit(min(limit, 500))
    return list(session.exec(query))


# ------------------------------------------------------------------------ decisions


@app.post("/api/decisions", response_model=Decision, status_code=201)
def create_decision(
    payload: DecisionCreate, session: Session = Depends(get_session)
) -> Decision:
    part_id = None
    if payload.part_serial:
        part_id = _part_by_serial(session, payload.part_serial).id

    decision = Decision(
        part_id=part_id,
        action=payload.action,
        rationale=payload.rationale,
        evidence=payload.evidence,
        model=payload.model,
        latency_ms=payload.latency_ms,
    )
    session.add(decision)
    session.commit()
    session.refresh(decision)
    return decision


@app.get("/api/decisions", response_model=list[Decision])
def list_decisions(limit: int = 50, session: Session = Depends(get_session)) -> list[Decision]:
    query = select(Decision).order_by(Decision.id.desc()).limit(min(limit, 500))
    return list(session.exec(query))


# ---------------------------------------------------------------------------- stats


@app.get("/api/stats", response_model=CellStats)
def cell_stats(
    window: int = 20,
    scored: int = 100,
    session: Session = Depends(get_session),
) -> CellStats:
    """Everything the dashboard and the drift check need, in one round trip."""

    parts_total = session.exec(select(func.count()).select_from(Part)).one()
    parts_accepted = session.exec(
        select(func.count()).select_from(Part).where(Part.status == PartStatus.ACCEPTED)
    ).one()
    parts_rework = session.exec(
        select(func.count()).select_from(Part).where(Part.status == PartStatus.REWORK)
    ).one()

    # Correction accuracy, ground-truthed rows only.
    errors = list(
        session.exec(
            select(Measurement.error_mm)
            .where(Measurement.error_mm.is_not(None))
            .order_by(Measurement.id.desc())
            .limit(scored)
        )
    )
    if errors:
        correction = ErrorStats(
            n=len(errors),
            mean_mm=statistics.fmean(errors),
            median_mm=statistics.median(errors),
            p95_mm=_percentile(errors, 95),
            max_mm=max(errors),
            sub_mm_rate=sum(1 for e in errors if e < 1.0) / len(errors),
        )
    else:
        correction = ErrorStats(n=0)

    # Inspection accuracy, rows where a defect was injected so truth is known.
    graded = list(
        session.exec(
            select(Inspection)
            .where(Inspection.correct.is_not(None))
            .order_by(Inspection.id.desc())
            .limit(scored)
        )
    )
    stats = InspectionStats(n_scored=len(graded))
    for row in graded:
        clean = row.injected_defect is not None and row.injected_defect.value == "none"
        called_pass = row.verdict == Verdict.PASS
        if clean and called_pass:
            stats.true_pass += 1
        elif clean and not called_pass:
            stats.false_defect += 1
        elif not clean and called_pass:
            stats.false_pass += 1
        else:
            stats.true_defect += 1
    if graded:
        stats.accuracy = (stats.true_pass + stats.true_defect) / len(graded)

    # Drift: is the recent window sitting off the earlier baseline?
    lateral = list(
        session.exec(
            select(Measurement.detected_y).order_by(Measurement.id.desc()).limit(scored)
        )
    )
    drift = DriftSignal(window=window)
    if len(lateral) >= window * 2:
        recent = statistics.fmean(lateral[:window]) * 1000.0
        baseline = statistics.fmean(lateral[window:]) * 1000.0
        drift.recent_mean_y_mm = recent
        drift.baseline_mean_y_mm = baseline
        drift.shift_mm = recent - baseline
        drift.trending = abs(recent - baseline) > 1.0

    return CellStats(
        generated_at=utcnow(),
        parts_total=parts_total,
        parts_accepted=parts_accepted,
        parts_rework=parts_rework,
        correction=correction,
        inspection=stats,
        drift=drift,
    )


# ------------------------------------------------------------------------ dashboard

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
