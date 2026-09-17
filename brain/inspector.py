"""Post-weld inspection via a cloud vision model.

The model backend is pluggable on purpose. Token Factory serves no NVIDIA vision
model today (see docs/nebius.md), so the default is a non-NVIDIA one. If an NVIDIA
VL model is later self-hosted on Nebius AI Cloud, only the model id and base URL
change, not this code.

A finding worth keeping: the model can see a bead discontinuity perfectly well when
asked to describe the image, but will call the weld a pass if the prompt does not
define what a defect is. The defect definitions below are load-bearing, not decoration.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from brain import nebius
from service.models import DefectType, Verdict

SYSTEM_PROMPT = """You are a weld inspector examining a single fillet weld bead \
photographed after welding.

Judge only what is visible in the image. Do not speculate about causes.

Defect definitions:
- skip: the bead stops and restarts, leaving a length of joint with no weld metal. \
Any visible break in bead continuity is a skip.
- undercut: a groove melted into the parent metal along the bead toe, leaving a notch \
that reduces the throat.
- porosity: gas pores, visible as small dark rounded cavities in the bead face.
- spatter: loose droplets of metal scattered on the parent metal beside the bead.
- burn_through: a hole melted fully through the joint.

A bead is a pass ONLY if it runs continuously end to end and shows none of the above.
Report the single most severe defect visible. If the image is too dark, blurred or \
cropped to judge, return the verdict "uncertain"."""

USER_PROMPT = (
    "Inspect this weld. Reply with JSON only, no prose and no code fence:\n"
    '{"verdict": "pass" | "defect" | "uncertain", '
    '"defect_type": "none" | "undercut" | "skip" | "porosity" | "spatter" | "burn_through", '
    '"severity": 0.0 to 1.0, '
    '"reasoning": "one sentence citing what you saw"}'
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass
class InspectionResult:
    verdict: Verdict
    defect_type: DefectType
    severity: Optional[float] = None
    reasoning: Optional[str] = None
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    raw: str = ""
    parse_error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return self.verdict == Verdict.PASS

    def as_payload(self, part_serial: str, image_path: str) -> dict[str, Any]:
        """Body for POST /api/inspections."""
        return {
            "part_serial": part_serial,
            "image_path": image_path,
            "verdict": self.verdict.value,
            "defect_type": self.defect_type.value,
            "severity": self.severity,
            "reasoning": self.reasoning,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of a reply that may carry fences or stray prose."""
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


def _coerce(data: dict[str, Any]) -> tuple[Verdict, DefectType, Optional[float]]:
    try:
        verdict = Verdict(str(data.get("verdict", "")).strip().lower())
    except ValueError:
        verdict = Verdict.UNCERTAIN

    try:
        defect = DefectType(str(data.get("defect_type", "none")).strip().lower())
    except ValueError:
        defect = DefectType.NONE

    # Keep the two fields consistent; the model occasionally disagrees with itself.
    if verdict == Verdict.PASS:
        defect = DefectType.NONE
    elif verdict == Verdict.DEFECT and defect == DefectType.NONE:
        verdict = Verdict.UNCERTAIN

    severity = data.get("severity")
    try:
        severity = min(1.0, max(0.0, float(severity))) if severity is not None else None
    except (TypeError, ValueError):
        severity = None

    return verdict, defect, severity


def inspect_weld(
    image_path: str | Path,
    model: Optional[str] = None,
    detail: str = "high",
    temperature: float = 0.0,
    max_tokens: int = 400,
) -> InspectionResult:
    """Send one post-weld image for a verdict.

    Raises on transport failure. A malformed reply is returned as an uncertain
    verdict with parse_error set, so one bad response cannot stall a campaign.
    """
    model = model or nebius.vision_model()
    client = nebius.client()

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    nebius.image_part(image_path, detail=detail),
                ],
            },
        ],
    )
    latency_ms = (time.perf_counter() - started) * 1000

    raw = nebius.reply_text(response.choices[0].message)
    usage = response.usage

    common = {
        "model": model,
        "latency_ms": latency_ms,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "raw": raw,
    }

    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return InspectionResult(
            verdict=Verdict.UNCERTAIN,
            defect_type=DefectType.NONE,
            reasoning="Model reply could not be parsed.",
            parse_error=f"{type(exc).__name__}: {exc}",
            **common,
        )

    verdict, defect, severity = _coerce(data)
    reasoning = str(data.get("reasoning") or "").strip() or None

    return InspectionResult(
        verdict=verdict,
        defect_type=defect,
        severity=severity,
        reasoning=reasoning,
        **common,
    )
