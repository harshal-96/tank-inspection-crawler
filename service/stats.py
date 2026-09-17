"""Derived numbers for the cell.

Shared by the HTTP API and the supervisor agent's tools so the two can never
disagree about what the line is doing. Arithmetic lives here, deliberately, and
not inside the language model.
"""

from __future__ import annotations

import statistics
from typing import Optional

from sqlmodel import Session, func, select

from service.models import (
    Inspection,
    Measurement,
    Part,
    PartStatus,
    Verdict,
    utcnow,
)
from service.schemas import CellStats, DriftSignal, ErrorStats, InspectionStats

# How many standard errors a window-to-window shift must clear before it is called
# a trend. Two sigma is the usual bar for "this is not scatter". Judging drift by
# millimetres alone fires constantly, because parts arrive across a band tens of
# millimetres wide and two twenty-part windows differ by several millimetres on
# chance alone.
DRIFT_SIGMA_THRESHOLD = 2.0

# Correction error above this on a single part is an action, not a trend.
ERROR_ACTION_THRESHOLD_MM = 3.0


def error_mm(
    detected_x: float, detected_y: float, true_x: Optional[float], true_y: Optional[float]
) -> Optional[float]:
    """Planar distance between the detected joint and the real one, in millimetres."""
    if true_x is None or true_y is None:
        return None
    return 1000.0 * ((detected_x - true_x) ** 2 + (detected_y - true_y) ** 2) ** 0.5


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. Stable for the small n a shift produces."""
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(pct / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[idx]


def correction_stats(session: Session, limit: int = 100) -> ErrorStats:
    errors = list(
        session.exec(
            select(Measurement.error_mm)
            .where(Measurement.error_mm.is_not(None))
            .order_by(Measurement.id.desc())
            .limit(limit)
        )
    )
    if not errors:
        return ErrorStats(n=0)
    return ErrorStats(
        n=len(errors),
        mean_mm=statistics.fmean(errors),
        median_mm=statistics.median(errors),
        p95_mm=percentile(errors, 95),
        max_mm=max(errors),
        sub_mm_rate=sum(1 for e in errors if e < 1.0) / len(errors),
        over_3mm=sum(1 for e in errors if e > ERROR_ACTION_THRESHOLD_MM),
    )


def inspection_stats(session: Session, limit: int = 100) -> InspectionStats:
    """Confusion matrix against injected defects. Unscored rows are ignored."""
    graded = list(
        session.exec(
            select(Inspection)
            .where(Inspection.correct.is_not(None))
            .order_by(Inspection.id.desc())
            .limit(limit)
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
    return stats


def drift_signal(session: Session, window: int = 20, limit: int = 100) -> DriftSignal:
    """Compare the recent window's lateral mean against the parts before it.

    A single part cannot show this. A locator wearing, or a fixture settling,
    moves every part the same way, which only appears as a trend.
    """
    lateral = list(
        session.exec(select(Measurement.detected_y).order_by(Measurement.id.desc()).limit(limit))
    )
    signal = DriftSignal(window=window)
    if len(lateral) < window * 2:
        return signal

    recent_values = [v * 1000.0 for v in lateral[:window]]
    baseline_values = [v * 1000.0 for v in lateral[window:]]
    recent = statistics.fmean(recent_values)
    baseline = statistics.fmean(baseline_values)
    shift = recent - baseline

    spread = statistics.pstdev([v * 1000.0 for v in lateral])
    noise = spread * (1.0 / len(recent_values) + 1.0 / len(baseline_values)) ** 0.5

    signal.recent_mean_y_mm = recent
    signal.baseline_mean_y_mm = baseline
    signal.shift_mm = shift
    signal.spread_mm = spread
    signal.noise_mm = noise
    signal.sigma = abs(shift) / noise if noise > 1e-9 else 0.0
    signal.trending = signal.sigma > DRIFT_SIGMA_THRESHOLD
    return signal


def compute_stats(session: Session, window: int = 20, scored: int = 100) -> CellStats:
    """Everything the dashboard and the agent need, in one pass."""
    return CellStats(
        generated_at=utcnow(),
        parts_total=session.exec(select(func.count()).select_from(Part)).one(),
        parts_accepted=session.exec(
            select(func.count()).select_from(Part).where(Part.status == PartStatus.ACCEPTED)
        ).one(),
        parts_rework=session.exec(
            select(func.count()).select_from(Part).where(Part.status == PartStatus.REWORK)
        ).one(),
        correction=correction_stats(session, scored),
        inspection=inspection_stats(session, scored),
        drift=drift_signal(session, window, scored),
    )
