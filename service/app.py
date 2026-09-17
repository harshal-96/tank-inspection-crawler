"""FastAPI service for the production cell.

Single source of truth. The simulation writes to it, the dashboard reads from it,
and the supervisor agent's tools query it. Nothing keeps its own copy of the state.
"""

from __future__ import annotations

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
    AgentMessage,
    Decision,
    Inspection,
    Measurement,
    Part,
    PartStatus,
    ShiftReport,
    Station,
    Verdict,
)
from service.schemas import (
    CellStats,
    ChatRequest,
    ChatResponse,
    DecisionCreate,
    InspectionCreate,
    MeasurementCreate,
    PartCreate,
)
from service.stats import compute_stats, error_mm

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
        error_mm=error_mm(
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
    return compute_stats(session, window=window, scored=scored)


# -------------------------------------------------------------------- supervisor


def _recent_history(session: Session, limit: int) -> list[dict[str, str]]:
    """Prior turns, oldest first. Tool traffic is left out; the agent re-queries."""
    rows = list(
        session.exec(
            select(AgentMessage)
            .where(AgentMessage.role.in_(["user", "assistant"]))
            .order_by(AgentMessage.id.desc())
            .limit(max(0, min(limit, 40)))
        )
    )
    return [{"role": r.role, "content": r.content or ""} for r in reversed(rows)]


@app.post("/api/agent/chat", response_model=ChatResponse)
def agent_chat(payload: ChatRequest, session: Session = Depends(get_session)) -> ChatResponse:
    """Ask the cell supervisor a question."""
    # Imported here so the service still starts without cloud credentials.
    from brain.nebius import MissingCredentials
    from brain.supervisor import ask

    history = _recent_history(session, payload.history_limit)
    session.add(AgentMessage(role="user", content=payload.message))
    session.commit()

    try:
        result = ask(payload.message, session, history=history)
    except MissingCredentials as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Supervisor unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    calls = [c.as_dict() for c in result.tool_calls]
    session.add(
        AgentMessage(
            role="assistant",
            content=result.reply,
            tool_calls=calls or None,
            model=result.model,
        )
    )
    session.commit()

    return ChatResponse(
        reply=result.reply,
        tool_calls=calls,
        model=result.model,
        latency_ms=result.latency_ms,
    )


@app.get("/api/agent/messages", response_model=list[AgentMessage])
def agent_messages(limit: int = 50, session: Session = Depends(get_session)) -> list[AgentMessage]:
    query = select(AgentMessage).order_by(AgentMessage.id.desc()).limit(min(limit, 200))
    return list(reversed(list(session.exec(query))))


@app.post("/api/agent/report", response_model=ShiftReport, status_code=201)
def agent_report(session: Session = Depends(get_session)) -> ShiftReport:
    """Generate and store an end-of-shift report."""
    from brain.nebius import MissingCredentials
    from brain.supervisor import shift_report

    first = session.exec(select(Part).order_by(Part.id.asc())).first()
    last = session.exec(select(Part).order_by(Part.id.desc())).first()
    if first is None or last is None:
        raise HTTPException(status_code=409, detail="No parts on record to report on")

    try:
        result = shift_report(session)
    except MissingCredentials as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    report = ShiftReport(
        period_start=first.created_at,
        period_end=last.created_at,
        content_md=result.reply,
        model=result.model,
    )
    session.add(report)
    session.commit()
    session.refresh(report)
    return report


@app.get("/api/agent/reports", response_model=list[ShiftReport])
def agent_reports(limit: int = 10, session: Session = Depends(get_session)) -> list[ShiftReport]:
    query = select(ShiftReport).order_by(ShiftReport.id.desc()).limit(min(limit, 50))
    return list(session.exec(query))


# ------------------------------------------------------------------------ dashboard

if DASHBOARD_DIR.exists():
    app.mount("/", StaticFiles(directory=DASHBOARD_DIR, html=True), name="dashboard")
