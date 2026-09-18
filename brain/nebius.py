"""Client for Nebius Token Factory.

Token Factory speaks the OpenAI wire protocol, so the official SDK works against
it with a swapped base URL. Everything provider-specific is isolated here.

Auth note: Token Factory API keys are NOT the same credential as a Nebius AI
Cloud console user_id/org_id. Create a key at tokenfactory.nebius.com under your
project.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Verified against the live catalog at
# https://tokenfactory.nebius.com/proxy/inference/private/v1/models_info
# Both are serverless (no dedicated-endpoint tag) and support tool calling.
#
# There is currently NO NVIDIA vision model on Token Factory. NVIDIA's multimodal
# models were withdrawn from serverless on 2026-08-31 and the replacement offered
# for Nemotron-3-Nano-Omni is text-only. The vision default below is therefore a
# non-NVIDIA model. See docs/nebius.md.
DEFAULT_TEXT_MODEL = "nvidia/Nemotron-3_5-Lightning"
# On the same fifty beads, under a prompt that tells the model the picture is
# a rendering, MiniMax-M3, DeepSeek V4.1 Flash and Gemma 3 27B tie within noise.
# MiniMax names the most defects right with no false alarms at 2.5 s a bead.
DEFAULT_VISION_MODEL = "MiniMaxAI/MiniMax-M3"

# The OpenAPI spec caps the image_url string at this length.
MAX_IMAGE_URL_CHARS = 13_981_514

# Accepted names for the Token Factory key, in priority order.
API_KEY_VARS = ("NEBIUS_API_KEY", "INFERENCE_API_KEY", "NEBIUS_INFERENCE_API_KEY")

_MISSING_KEY_HELP = """No Token Factory API key found.
Set one of: {names}
  1. Open https://tokenfactory.nebius.com and select your project
  2. Create an API key under API keys
  3. Add it to .env
Note: a Nebius AI Cloud user_id/org_id will not work here, it is a different
product with different auth."""


class MissingCredentials(RuntimeError):
    """Raised when no API key is set, with instructions rather than a stack trace."""


def api_key() -> str:
    for var in API_KEY_VARS:
        value = os.getenv(var, "").strip()
        if value:
            return value
    raise MissingCredentials(_MISSING_KEY_HELP.format(names=", ".join(API_KEY_VARS)))


def client() -> OpenAI:
    """An OpenAI SDK client pointed at Token Factory."""
    return OpenAI(
        api_key=api_key(),
        base_url=os.getenv("NEBIUS_BASE_URL", "").strip() or DEFAULT_BASE_URL,
    )


def text_model() -> str:
    return os.getenv("NEMOTRON_TEXT_MODEL", "").strip() or DEFAULT_TEXT_MODEL


def vision_model() -> str:
    return os.getenv("NEMOTRON_VISION_MODEL", "").strip() or DEFAULT_VISION_MODEL


def encode_image(path: str | Path) -> str:
    """Read an image into a data URL suitable for an image_url content part."""
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(suffix)
    if mime is None:
        raise ValueError(f"Unsupported image type {path.suffix!r}, use png, jpg or webp")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    url = f"data:image/{mime};base64,{encoded}"
    if len(url) > MAX_IMAGE_URL_CHARS:
        raise ValueError(
            f"{path.name} encodes to {len(url):,} characters, over the "
            f"{MAX_IMAGE_URL_CHARS:,} limit. Downscale it before sending."
        )
    return url


def image_part(path: str | Path, detail: str = "auto") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": encode_image(path), "detail": detail}}


def image_part_from_bytes(data: bytes, mime: str = "png", detail: str = "auto") -> dict[str, Any]:
    """The same content part for an image made in memory, such as a crop."""
    url = f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"
    if len(url) > MAX_IMAGE_URL_CHARS:
        raise ValueError("image too large to send inline")
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def strip_reasoning(content: Optional[str]) -> str:
    """Drop a leading <think>...</think> block if the model emits one inline."""
    text = (content or "").strip()
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def reply_text(message: Any) -> str:
    """The model's answer, from wherever that model decided to put it.

    The Nemotron sizes do not agree on this, and the disagreement is silent.
    Lightning fills `content` and leaves `reasoning` empty. Nano does the exact
    reverse: `content` comes back None and the answer, JSON and all, arrives in
    `reasoning`.

    Reading only `content` therefore does not raise or warn on Nano, it returns
    an empty string, and any caller with a sensible fallback for an unparseable
    reply quietly applies that fallback to every single call. An evaluation built
    on that measures the fallback and reports it as the model's judgement, which
    is exactly what happened here before this function existed: Nano appeared to
    score perfectly on a decision it was never asked.

    So the answer is taken from either field, and the caller does not need to know
    which model it is talking to.
    """
    text = strip_reasoning(getattr(message, "content", None))
    if text:
        return text
    return strip_reasoning(getattr(message, "reasoning", None))


def available_models() -> list[str]:
    """Model IDs this key can actually call. The only reliable check."""
    return sorted(m.id for m in client().models.list().data)
