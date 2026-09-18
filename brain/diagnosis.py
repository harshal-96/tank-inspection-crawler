"""Once a shift: which machine is going out of adjustment, from the seams the cell tracked.

The hard-case study scored six models on this question and picked one. This is
the code path that asks it in production, which did not exist before: the
platform page said "once per shift" and nothing ran. The rows are built from the
seam tracks the cell already stores, one per part, the bottom and top of each
seam as an offset from where a perfect part would put it. The model is asked the
same question the study asked, with the fitted slopes and the boundary sentence,
and the arithmetic rule answers beside it so an operator can see when the two
disagree.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from brain import nebius
from cell import geometry as geo
from cell.drift import (
    THRESHOLD_NOTE,
    discriminators,
    parse_verdict,
    prompt_for,
    rule,
    stats_block,
    user_prompt,
)

# Fewer parts than this and a slope against its own scatter means nothing.
MIN_PARTS = 8


def rows_from_tracks(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One diagnosis row per part from stored seam tracks, oldest first.

    Each track's path holds stations along the seam with the joint's measured
    y in millimetres. The bottom station is the first along the seam and the
    top the last, and both are taken as offsets from the nominal joint, which
    is where the taught path expects it.
    """
    nominal_y_mm = geo.nominal_joint()[1] * 1000.0
    rows = []
    for index, track in enumerate(tracks):
        path = [p for p in (track.get("path") or []) if p.get("y_mm") is not None]
        if len(path) < 2:
            continue
        path = sorted(path, key=lambda p: p.get("along_mm", 0.0))
        bottom = float(path[0]["y_mm"]) - nominal_y_mm
        top = float(path[-1]["y_mm"]) - nominal_y_mm
        rows.append({
            "part": index,
            "seam_bottom_mm": bottom,
            "seam_top_mm": top,
            "tilt_mm": top - bottom,
        })
    return rows


def diagnose(rows: list[dict[str, Any]], model: Optional[str] = None,
             api: Optional[Any] = None) -> dict[str, Any]:
    """Ask the diagnosis model the study's question, and the rule beside it."""
    if len(rows) < MIN_PARTS:
        return {"available": False,
                "reason": f"Needs at least {MIN_PARTS} tracked parts, have {len(rows)}."}
    model = model or nebius.diagnosis_model()
    messages = [
        {"role": "system", "content": prompt_for(rows) + THRESHOLD_NOTE},
        {"role": "user", "content": user_prompt(rows) + stats_block(rows)},
    ]
    started = time.perf_counter()
    reply = nebius.chat(messages, model=model, max_tokens=500, reasoning_effort="none",
                        parses=nebius.extract_json, api=api)
    verdict = parse_verdict(reply.raw or "", allowed=("press", "fixture", "none"))
    ratios = discriminators(rows)
    return {
        "available": True,
        "parts": len(rows),
        "cause": verdict.get("cause"),
        "confidence": verdict.get("confidence"),
        "evidence": verdict.get("evidence"),
        "rule_cause": rule(rows),
        "agree": verdict.get("cause") == rule(rows),
        "tilt_slope_mm_per_part": round(ratios["tilt_slope"], 4),
        "tilt_times_scatter": round(ratios["tilt_ratio"], 2),
        "shift_slope_mm_per_part": round(ratios["shift_slope"], 4),
        "shift_times_scatter": round(ratios["shift_ratio"], 2),
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "retried_with_room": reply.retried_with_room,
    }
