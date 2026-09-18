"""Client for Nebius Token Factory.

Token Factory speaks the OpenAI wire protocol, so the official SDK works against
it with a swapped base URL. Everything provider-specific is isolated here.

Auth note: Token Factory API keys are NOT the same credential as a Nebius AI
Cloud console user_id/org_id. Create a key at tokenfactory.nebius.com under your
project.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
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
# On 50 beads MiniMax-M3, DeepSeek V4.1 Flash and Gemma 3 27B tied within
# noise. On 150, on the pass-or-defect call the cell routes on, DeepSeek is
# right on 6 beads where MiniMax is wrong and never the reverse (p 0.031), and
# ships 14 bad parts to MiniMax's 20. MiniMax still names the defect best.
DEFAULT_VISION_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

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


# The SDK's own defaults are a 600 second timeout and two transport retries. A
# cell waiting ten minutes on one verdict is a stopped line, so the timeout is
# set here and can be tuned per site without touching code.
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_RETRIES = 3


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    return float(value) if value else default


def client() -> OpenAI:
    """An OpenAI SDK client pointed at Token Factory."""
    return OpenAI(
        api_key=api_key(),
        base_url=os.getenv("NEBIUS_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        timeout=_env_float("NEBIUS_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        max_retries=int(_env_float("NEBIUS_MAX_RETRIES", DEFAULT_MAX_RETRIES)),
    )


def text_model() -> str:
    return os.getenv("NEMOTRON_TEXT_MODEL", "").strip() or DEFAULT_TEXT_MODEL


# The once-a-shift question, which machine is going out of adjustment, on the
# 64 boundary cases: Qwen3.5 397B 46 of 64, DeepSeek V4.1 Flash 45, Ultra 45,
# Super 120B 44. Only the first three beat the arithmetic rule at p < 0.05.
# DeepSeek V4.1 Flash is the cheapest of them, so it is the default.
DEFAULT_DIAGNOSIS_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


def diagnosis_model() -> str:
    return os.getenv("CELL_DIAGNOSIS_MODEL", "").strip() or DEFAULT_DIAGNOSIS_MODEL


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



# ------------------------------------------------------------- one call policy
#
# Every model the cell talks to is called through chat(). Two things went wrong
# before this existed, each of them silently, each of them making a strong model
# look weak:
#
#   - reasoning_effort="none" is what makes Nemotron answer instead of think
#     aloud, and some other models reject the option with a 400. The option is
#     the envelope, not the question, so on that error it is dropped and the
#     call is made again.
#   - a reasoning model spends its output budget thinking before it answers,
#     and under a small max_tokens it comes back with nothing in the answer and
#     a full reasoning field. The parser reads that as a failure and a study
#     scores it as a wrong verdict. So a reply that does not parse, when the
#     budget was spent thinking, is asked once more with room.
#
# What a model is asked never changes here; only how the call is made. Which
# path a call took is recorded so a report can say so.

THINKING_TOKENS = 3000

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_json(text: str) -> dict[str, Any]:
    """The JSON object in a reply that may carry fences or stray prose.

    Raises ValueError or json.JSONDecodeError when there is none.
    """
    cleaned = _FENCE.sub("", (text or "").strip()).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, depth = cleaned.find("{"), 0
    if start == -1:
        raise ValueError("no JSON object in reply")
    for i, ch in enumerate(cleaned[start:], start):
        depth += (ch == "{") - (ch == "}")
        if depth == 0:
            data = json.loads(cleaned[start:i + 1])
            if not isinstance(data, dict):
                raise ValueError("reply JSON is not an object")
            return data
    raise ValueError("unbalanced JSON object in reply")


@dataclass
class ChatReply:
    raw: str
    model: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    retried_with_room: bool = False
    dropped_option: Optional[str] = None
    response: Any = None

    @property
    def effort(self) -> str:
        """Which envelope the answer came from, for a report."""
        return "default" if self.dropped_option else "none"


def _spent_thinking(response: Any, raw: str, cap: int, parses) -> bool:
    """Did the model think instead of answering, rather than answer badly?"""
    try:
        parses(raw)
        return False
    except Exception:  # noqa: BLE001  any parse failure counts
        pass
    message = response.choices[0].message
    thought = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    usage = response.usage
    exhausted = bool(usage and usage.completion_tokens is not None
                     and usage.completion_tokens >= cap)
    return bool(thought) or exhausted


def chat(
    messages: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
    max_tokens: int = 400,
    temperature: float = 0.0,
    reasoning_effort: Optional[str] = None,
    parses=None,
    thinking_tokens: int = THINKING_TOKENS,
    api: Optional[OpenAI] = None,
    **options: Any,
) -> ChatReply:
    """One chat completion under the policy above.

    parses is a callable that raises when the reply is not usable; when given,
    an unusable reply that spent its budget thinking is retried once at
    thinking_tokens. Transport failures raise.
    """
    from openai import BadRequestError

    api = api or client()
    model = model or text_model()
    started = time.perf_counter()

    def call(tokens: int, effort: Optional[str]) -> Any:
        extra = {"reasoning_effort": effort} if effort else {}
        return api.chat.completions.create(
            model=model, temperature=temperature, max_tokens=tokens,
            messages=messages, **options, **extra)

    dropped = None
    effort = reasoning_effort
    try:
        response = call(max_tokens, effort)
    except BadRequestError as exc:
        if effort and "reasoning" in str(exc).lower():
            dropped, effort = "reasoning_effort", None
            response = call(max_tokens, effort)
        else:
            raise
    raw = reply_text(response.choices[0].message)

    retried = False
    if (parses is not None and max_tokens < thinking_tokens
            and _spent_thinking(response, raw, max_tokens, parses)):
        response = call(thinking_tokens, effort)
        raw = reply_text(response.choices[0].message)
        retried = True

    usage = response.usage
    return ChatReply(
        raw=raw, model=model, latency_ms=(time.perf_counter() - started) * 1000,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None,
        retried_with_room=retried, dropped_option=dropped, response=response,
    )
