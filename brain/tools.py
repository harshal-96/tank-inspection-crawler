"""Tools the supervisor agent can call.

Design rule: tools return facts, the model supplies judgement. Every number the
agent quotes comes from a query here, never from its own arithmetic, so a
hallucinated statistic cannot reach the operator.

The one write tool records a decision as a recommendation with acknowledged=False.
Nothing here moves a robot.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from sqlmodel import Session, select

from service.models import Action, Decision, Inspection, Measurement, Part, Verdict
from service.stats import compute_stats, drift_signal

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_cell_status",
            "description": (
                "Current state of the whole cell: part counts, correction error "
                "statistics in millimetres, inspection accuracy against injected "
                "defects, and the lateral drift signal. Call this first for any "
                "general question about how the line is running. Counts are exact "
                "counts over n parts. Do not derive additional counts or percentages "
                "from p95 or max, those describe the distribution only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {
                        "type": "integer",
                        "description": "Parts in the recent drift window, default 20",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_parts",
            "description": (
                "The last N parts with their measured correction error, detected "
                "position and inspection verdict. Use when asked about trends or "
                "about what just happened on the line."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many parts, default 10, max 50"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_part_detail",
            "description": "Everything known about one part, by serial, for example P-000042.",
            "parameters": {
                "type": "object",
                "properties": {"serial": {"type": "string"}},
                "required": ["serial"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_drift",
            "description": (
                "Lateral drift detail: the recent window mean against the earlier "
                "baseline, in millimetres. Positive means the joint is being found "
                "further in +y than before. Judge it by sigma, not millimetres: "
                "parts arrive across a wide band, so a shift under 2 sigma is "
                "ordinary scatter and means nothing. Sustained drift above that "
                "suggests a mechanical cause such as locator wear."
            ),
            "parameters": {
                "type": "object",
                "properties": {"window": {"type": "integer", "description": "Default 20"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_defects",
            "description": "Recent parts the inspection model flagged as defective.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "description": "Default 10"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_decision",
            "description": (
                "Record a recommendation for the operator. Use only when you have "
                "evidence from another tool. The decision is logged for acknowledgement, "
                "it does not act on the cell by itself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["accept", "rework", "recalibrate", "hold_line"],
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One or two sentences citing the evidence.",
                    },
                    "part_serial": {
                        "type": "string",
                        "description": "Only if the decision concerns a single part.",
                    },
                },
                "required": ["action", "rationale"],
            },
        },
    },
]


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def _part_summary(session: Session, part: Part) -> dict[str, Any]:
    measurement = session.exec(
        select(Measurement).where(Measurement.part_id == part.id).order_by(Measurement.id.desc())
    ).first()
    inspection = session.exec(
        select(Inspection).where(Inspection.part_id == part.id).order_by(Inspection.id.desc())
    ).first()

    summary: dict[str, Any] = {"serial": part.serial, "status": part.status.value}
    if measurement:
        summary |= {
            "error_mm": _round(measurement.error_mm),
            "detected_y_mm": _round(measurement.detected_y * 1000.0, 1),
            "confidence": _round(measurement.confidence),
        }
    if inspection:
        summary |= {
            "verdict": inspection.verdict.value,
            "defect_type": inspection.defect_type.value,
        }
    return summary


# --------------------------------------------------------------------------- tools


def get_cell_status(session: Session, window: int = 20) -> dict[str, Any]:
    stats = compute_stats(session, window=window)
    return {
        "parts_total": stats.parts_total,
        "parts_accepted": stats.parts_accepted,
        "parts_rework": stats.parts_rework,
        "correction_error_mm": {
            "n": stats.correction.n,
            "mean": _round(stats.correction.mean_mm),
            "median": _round(stats.correction.median_mm),
            "p95": _round(stats.correction.p95_mm),
            "max": _round(stats.correction.max_mm),
            "sub_mm_rate": _round(stats.correction.sub_mm_rate),
            "parts_over_3mm": stats.correction.over_3mm,
        },
        "inspection": {
            "n_scored": stats.inspection.n_scored,
            "accuracy": _round(stats.inspection.accuracy),
            "missed_defects": stats.inspection.false_pass,
            "false_alarms": stats.inspection.false_defect,
        },
        "drift": {
            "shift_mm": _round(stats.drift.shift_mm),
            "noise_mm": _round(stats.drift.noise_mm),
            "sigma": _round(stats.drift.sigma),
            "trending": stats.drift.trending,
            "window": stats.drift.window,
        },
    }


def get_recent_parts(session: Session, n: int = 10) -> dict[str, Any]:
    n = max(1, min(int(n), 50))
    parts = list(session.exec(select(Part).order_by(Part.id.desc()).limit(n)))
    return {"parts": [_part_summary(session, p) for p in parts]}


def get_part_detail(session: Session, serial: str) -> dict[str, Any]:
    part = session.exec(select(Part).where(Part.serial == serial)).first()
    if part is None:
        return {"error": f"No part with serial {serial}"}

    detail = _part_summary(session, part)
    detail |= {
        "station": part.station.value,
        "spawn_dy_mm": _round(part.spawn_dy_mm, 1),
        "spawn_yaw_deg": _round(part.spawn_yaw_deg, 2),
        "created_at": part.created_at.isoformat(),
    }
    inspection = session.exec(
        select(Inspection).where(Inspection.part_id == part.id).order_by(Inspection.id.desc())
    ).first()
    if inspection and inspection.reasoning:
        detail["inspection_reasoning"] = inspection.reasoning
    return detail


def get_drift(session: Session, window: int = 20) -> dict[str, Any]:
    signal = drift_signal(session, window=window)
    if signal.shift_mm is None:
        return {
            "available": False,
            "reason": f"Needs at least {signal.window * 2} measured parts to compare windows.",
        }
    return {
        "available": True,
        "window": signal.window,
        "recent_mean_y_mm": _round(signal.recent_mean_y_mm),
        "baseline_mean_y_mm": _round(signal.baseline_mean_y_mm),
        "shift_mm": _round(signal.shift_mm),
        "part_to_part_spread_mm": _round(signal.spread_mm),
        "noise_on_the_shift_mm": _round(signal.noise_mm),
        "sigma": _round(signal.sigma),
        "trending": signal.trending,
    }


def list_defects(session: Session, limit: int = 10) -> dict[str, Any]:
    limit = max(1, min(int(limit), 50))
    rows = list(
        session.exec(
            select(Inspection, Part)
            .join(Part, Inspection.part_id == Part.id)
            .where(Inspection.verdict == Verdict.DEFECT)
            .order_by(Inspection.id.desc())
            .limit(limit)
        )
    )
    return {
        "defects": [
            {
                "serial": part.serial,
                "defect_type": inspection.defect_type.value,
                "severity": _round(inspection.severity),
                "reasoning": inspection.reasoning,
            }
            for inspection, part in rows
        ]
    }


def record_decision(
    session: Session,
    action: str,
    rationale: str,
    part_serial: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    try:
        parsed = Action(action.strip().lower())
    except ValueError:
        return {"error": f"Unknown action {action!r}. Valid: {[a.value for a in Action]}"}

    part_id = None
    if part_serial:
        part = session.exec(select(Part).where(Part.serial == part_serial)).first()
        if part is None:
            return {"error": f"No part with serial {part_serial}"}
        part_id = part.id

    decision = Decision(
        part_id=part_id,
        action=parsed,
        rationale=rationale,
        evidence={"source": "supervisor_agent"},
        model=model,
    )
    session.add(decision)
    session.commit()
    session.refresh(decision)
    return {
        "recorded": True,
        "decision_id": decision.id,
        "action": parsed.value,
        "awaiting_acknowledgement": True,
    }


REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "get_cell_status": get_cell_status,
    "get_recent_parts": get_recent_parts,
    "get_part_detail": get_part_detail,
    "get_drift": get_drift,
    "list_defects": list_defects,
    "record_decision": record_decision,
}


def dispatch(
    name: str, arguments: dict[str, Any], session: Session, model: Optional[str] = None
) -> dict[str, Any]:
    """Run one tool call. Never raises: a failure is returned as data for the model."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name!r}"}

    kwargs = dict(arguments or {})
    if name == "record_decision":
        kwargs["model"] = model
    try:
        return fn(session, **kwargs)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
