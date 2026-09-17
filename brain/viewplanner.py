"""The cloud half of the look-and-refine loop.

Before the arc strikes, the cell measures the joint from one view, and the model
decides whether that measurement can be trusted or whether the part should be
presented differently and measured again. The robot and the cloud converge on a
weld path together.

This is not arithmetic dressed up. The geometry already produces the numbers; what
the model supplies is the judgement about whether a given look is good enough,
across signals that do not reduce to a single threshold, and an explanation an
operator can act on.

Runs on the NVIDIA reasoning model, so the cloud sits inside the control loop
rather than grading the result afterwards.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from brain import nebius

MAX_TOKENS = 1200

# Nemotron reasons at length by default and will spend an entire token budget
# thinking out loud in plain prose before it ever reaches an answer. Measured
# against this prompt: guided_json through extra_body is ignored, response_format
# json_object is ignored, and reasoning_effort "low" is not enough. Only
# reasoning_effort "none" produces the answer directly, and it cut the reply from
# 900 tokens of preamble to 86 tokens of correct JSON.
REASONING_EFFORT = "none"

SYSTEM_PROMPT = """You direct a robot measuring a weld joint before it welds.

The joint is the line where two panel faces meet. The robot photographs it with a
depth camera, fits a plane to each face and intersects them. It can present the
part at a different angle and measure again. Your job is to decide whether the
measurement in hand is trustworthy, and if not, which angle to try next.

What the signals mean.
- included_angle_deg is the angle between the two fitted faces. These panels meet
  square, so a healthy view reads near 90. Well away from 90 means the fit is
  being distorted, usually because one face is nearly edge-on.
- inlier_balance runs 0 to 1. Near 1 means both faces are seen about equally.
  Below roughly 0.35 one face is barely visible and the intersection is being
  driven by the other, which is the classic way a joint estimate goes wrong.
- face_a_fit_rms_mm and face_b_fit_rms_mm are how well each face matched a plane.
  Large values mean noise, curvature or contamination such as spatter.
- path_length_mm is how much of the joint this view covered. A short path means
  much of the weld was out of frame or occluded.
- Agreement between views is the strongest evidence there is. Two independent
  views landing in the same place is worth more than one view with tidy numbers.

How to decide.
- Accept when at least two views agree closely and at least one of them had good
  balance and a near-square angle. Accept a single view only if its signals are
  excellent and no other view is available.
- Otherwise choose an untried angle. Reason about which one helps: if a face is
  edge-on, swing far enough to open it up, not a few degrees. If two views already
  disagree, a third well away from both settles it.
- Never invent a number. Every figure you cite must appear in the data given.
- Never choose an angle that is not in the available list.

Reply with JSON only, no prose and no code fence:
{"accept": true|false, "next_view_deg": <number from the available list, or null>,
 "reason": "one or two sentences citing the signals"}"""

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class ViewDecision:
    accept: bool
    next_view_deg: Optional[float] = None
    reason: str = ""
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = ""
    parse_error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = _FENCE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, depth = cleaned.find("{"), 0
    if start == -1:
        raise ValueError("no JSON object in reply")
    for i, ch in enumerate(cleaned[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            return json.loads(cleaned[start : i + 1])
    raise ValueError("unbalanced JSON object in reply")


def spread_mm(attempts: list[dict[str, Any]]) -> Optional[float]:
    """Largest disagreement between the joint estimates so far, in millimetres."""
    points = [a["joint_mm"] for a in attempts if a.get("joint_mm")]
    if len(points) < 2:
        return None
    worst = 0.0
    for i, first in enumerate(points):
        for second in points[i + 1 :]:
            worst = max(worst, ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5)
    return worst


def build_prompt(attempts: list[dict[str, Any]], available_deg: list[float]) -> str:
    payload = {
        "views_measured": attempts,
        "angles_still_available_deg": available_deg,
        "largest_disagreement_mm": (
            None if spread_mm(attempts) is None else round(spread_mm(attempts), 3)
        ),
    }
    return (
        "Measurements so far, and the angles the robot has not yet tried:\n"
        + json.dumps(payload, indent=1)
        + "\n\nDecide."
    )


def plan_next_view(
    attempts: list[dict[str, Any]],
    available_deg: list[float],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> ViewDecision:
    """Accept the measurement, or name the next angle to try.

    Raises on transport failure. A malformed or out-of-range reply is returned as a
    decision to accept, because a planner that cannot answer must not be allowed to
    hold the line indefinitely.
    """
    model = model or nebius.text_model()
    client = nebius.client()

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(attempts, available_deg)},
        ],
    )
    latency_ms = (time.perf_counter() - started) * 1000
    raw = nebius.strip_reasoning(response.choices[0].message.content)
    usage = response.usage

    common = {
        "model": model,
        "latency_ms": latency_ms,
        "raw": raw,
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }

    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return ViewDecision(
            accept=True,
            reason="Planner reply could not be parsed; accepting what we have.",
            parse_error=f"{type(exc).__name__}: {exc}",
            **common,
        )

    accept = bool(data.get("accept", False))
    reason = str(data.get("reason") or "").strip()

    next_view = data.get("next_view_deg")
    try:
        next_view = None if next_view is None else float(next_view)
    except (TypeError, ValueError):
        next_view = None

    if not accept:
        if not available_deg:
            accept, next_view = True, None
            reason = (reason + " No further angles available.").strip()
        elif next_view is None or next_view not in available_deg:
            # Hold it to the list rather than letting it invent a pose.
            asked = next_view
            next_view = min(available_deg, key=lambda a: abs(a - (asked or 0.0)))
            reason = (
                f"{reason} (planner asked for {asked}, which is not available; "
                f"used {next_view} instead)"
            ).strip()

    return ViewDecision(accept=accept, next_view_deg=next_view, reason=reason, **common)
