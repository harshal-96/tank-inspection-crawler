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
import os
import re
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

# The prompt above calls the picture a photograph and offers "uncertain" as a way
# out. The bead pictures are renders from the simulator, and the stronger models
# say so and take the way out: "a computer-generated rendering, impossible to
# verify". That is the prompt's fault. This framing tells the model what it is
# looking at and closes the exit, and it is offered to every model the same way.
RENDER_PROMPT = SYSTEM_PROMPT.replace(
    "photographed after welding.",
    "as pictured after welding.\n\n"
    "The picture is a rendering from the welding simulator this cell is tested against, "
    "not a photograph. The bead is drawn as a raised strip along the vertical joint between "
    "two plates, and each defect is rendered into it with the shape described below. Judge "
    "the rendering exactly as you would a photograph. That the image is synthetic, flat or "
    "low in texture is not a reason to answer uncertain; answer uncertain only if no bead "
    "is in the frame.",
)

# On a real cell the bead is a photograph, and the honest framing says so. It
# closes the same exit the rendering prompt closes: an unfamiliar camera or an
# odd finish is not a reason to decline.
PHOTO_PROMPT = SYSTEM_PROMPT.replace(
    'If the image is too dark, blurred or cropped to judge, return the verdict "uncertain".',
    'Answer "uncertain" only if no bead is in the frame or the bead cannot be made out at '
    "all. Low contrast, an unfamiliar camera or an unusual surface finish are not reasons "
    "to decline; judge what is there.",
)

FRAMINGS = {"as_written": SYSTEM_PROMPT, "rendering": RENDER_PROMPT, "photograph": PHOTO_PROMPT}

# What the bead camera is looking at, set per site: BEAD_SOURCE=rendering on the
# simulator this project ships with, BEAD_SOURCE=photograph on a cell.
DEFAULT_FRAMING = "rendering"


def default_framing() -> str:
    value = os.getenv("BEAD_SOURCE", "").strip().lower()
    return value if value in FRAMINGS else DEFAULT_FRAMING


# The simulator's camera frames the whole part and the bead is a strip about
# twenty pixels wide down the middle of a 640 pixel frame. This is the window a
# bead camera would be aimed at: the strip with the plates either side, doubled.
# Another camera sets BEAD_WINDOW=x0,y0,x1,y1 in pixels, or BEAD_WINDOW=none to
# send the frame as it comes.
BEAD_WINDOW = (200, 30, 440, 610)


def bead_window() -> Optional[tuple[int, int, int, int]]:
    """The crop box for this site, or None when the frame goes as it is."""
    value = os.getenv("BEAD_WINDOW", "").strip().lower()
    if not value:
        return BEAD_WINDOW
    if value == "none":
        return None
    parts = [int(float(v)) for v in value.replace(";", ",").split(",")]
    if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise ValueError("BEAD_WINDOW must be x0,y0,x1,y1 with x0<x1 and y0<y1, or none")
    return parts[0], parts[1], parts[2], parts[3]


def bead_crop(image_path: str | Path,
              window: Optional[tuple[int, int, int, int]] = None) -> bytes:
    """The bead and the plate either side of it, at twice the size, as PNG bytes."""
    from io import BytesIO

    from PIL import Image

    box = window or bead_window() or BEAD_WINDOW
    with Image.open(image_path) as im:
        frame = im.convert("RGB")
        # Clip to the frame so a smaller picture is cropped, not padded.
        box = (max(0, box[0]), max(0, box[1]), min(frame.width, box[2]), min(frame.height, box[3]))
        window = frame.crop(box)
        window = window.resize((window.width * 2, window.height * 2), Image.LANCZOS)
        buf = BytesIO()
        window.save(buf, format="PNG")
        return buf.getvalue()


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


THINKING_TOKENS = nebius.THINKING_TOKENS
_extract_json = nebius.extract_json


def inspect_weld(
    image_path: str | Path,
    model: Optional[str] = None,
    detail: str = "high",
    temperature: float = 0.0,
    max_tokens: int = 400,
    framing: Optional[str] = None,
    crop: Optional[bool] = None,
) -> InspectionResult:
    """Send one post-weld image for a verdict.

    Raises on transport failure. A malformed reply is returned as an uncertain
    verdict with parse_error set, so one bad response cannot stall a campaign.
    A reply that is empty because the model spent its budget reasoning is sent
    again with a larger one, and the result records that it was.

    framing and crop default from the environment (BEAD_SOURCE, BEAD_WINDOW),
    because they describe the camera, not the model. On the simulator that is
    the rendering prompt and the bead shown up close, under which every model
    did better on the same fifty beads; the as-written prompt is kept so the
    comparison can be re-run.
    """
    model = model or nebius.vision_model()
    framing = framing or default_framing()
    if framing not in FRAMINGS:
        raise ValueError(f"framing must be one of {sorted(FRAMINGS)}, not {framing!r}")
    window = bead_window()
    if crop is None:
        crop = window is not None
    picture = (nebius.image_part_from_bytes(bead_crop(image_path, window), "png", detail=detail)
               if crop else nebius.image_part(image_path, detail=detail))
    messages = [
        {"role": "system", "content": FRAMINGS[framing]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                picture,
            ],
        },
    ]

    reply = nebius.chat(messages, model=model, max_tokens=max_tokens,
                        temperature=temperature, parses=_extract_json)
    raw = reply.raw

    common = {
        "model": model,
        "latency_ms": reply.latency_ms,
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "raw": raw,
        "extra": {"retried_with_room": reply.retried_with_room, "framing": framing,
                  "crop": crop},
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
