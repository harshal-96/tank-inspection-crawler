"""Preflight for the cloud half of the cell.

Run this before anything else. It answers four questions in order, and stops at
the first one that fails:

  1. Does the API key authenticate?
  2. Which models can this key actually call?
  3. Does the Nemotron agent model do tool calling?
  4. Does the vision model accept an image and return a verdict?

    python scripts/hello_nemotron.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402

OK, BAD, INFO = "[ ok ]", "[fail]", "[ .. ]"


def _rule(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def check_key() -> bool:
    _rule("1. Credentials")
    try:
        key = nebius.api_key()
    except nebius.MissingCredentials as exc:
        print(f"{BAD} {exc}")
        return False
    print(f"{OK} NEBIUS_API_KEY present ({len(key)} chars), base {nebius.DEFAULT_BASE_URL}")
    return True


def check_catalog() -> bool:
    _rule("2. Models callable by this key")
    try:
        models = nebius.available_models()
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} models.list() failed: {type(exc).__name__}: {exc}")
        print("       A 401 here means the key is wrong or belongs to AI Cloud, not Token Factory.")
        return False

    want_text, want_vision = nebius.text_model(), nebius.vision_model()
    print(f"{INFO} {len(models)} models visible")

    nvidia = [m for m in models if m.lower().startswith("nvidia/")]
    print(f"{INFO} NVIDIA models: {', '.join(nvidia) if nvidia else 'none'}")

    ok = True
    for label, wanted in (("agent", want_text), ("vision", want_vision)):
        if wanted in models:
            print(f"{OK} {label} model available: {wanted}")
        else:
            print(f"{BAD} {label} model NOT in catalog: {wanted}")
            close = [m for m in models if wanted.split("/")[-1][:8].lower() in m.lower()]
            if close:
                print(f"       closest matches: {', '.join(close)}")
            ok = False
    return ok


def check_text() -> bool:
    _rule("3. Nemotron agent, text and tool calling")
    client, model = nebius.client(), nebius.text_model()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_correction_error",
                "description": "Mean joint correction error over the last N parts, in millimetres.",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer", "description": "How many parts"}},
                    "required": ["n"],
                },
            },
        }
    ]

    try:
        t0 = time.perf_counter()
        first = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You supervise a robotic welding cell. Use tools for any number you do not already have.",
                },
                {"role": "user", "content": "What is our mean correction error over the last 20 parts?"},
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=512,
        )
        dt = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} chat.completions failed: {type(exc).__name__}: {exc}")
        return False

    msg = first.choices[0].message
    calls = msg.tool_calls or []
    if not calls:
        print(f"{BAD} model answered without calling the tool ({dt:.0f} ms)")
        print(f"       said: {nebius.strip_reasoning(msg.content)[:160]}")
        return False

    call = calls[0]
    print(f"{OK} tool call in {dt:.0f} ms: {call.function.name}({call.function.arguments})")

    # Feed a result back so the full round trip is exercised.
    try:
        second = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You supervise a robotic welding cell."},
                {"role": "user", "content": "What is our mean correction error over the last 20 parts?"},
                msg.model_dump(exclude_none=True),
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({"mean_mm": 0.99, "p95_mm": 1.97, "n": 20}),
                },
            ],
            max_tokens=512,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} tool-result round trip failed: {type(exc).__name__}: {exc}")
        return False

    reply = nebius.strip_reasoning(second.choices[0].message.content)
    print(f"{OK} answered: {reply[:200]}")
    if second.usage:
        print(f"{INFO} tokens in/out: {second.usage.prompt_tokens}/{second.usage.completion_tokens}")
    return True


def _test_image(path: Path) -> Path:
    """A crude weld bead with an obvious gap, enough to prove the image path works."""
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (640, 240), (38, 38, 42))
    d = ImageDraw.Draw(img)
    for x in range(40, 600, 7):
        if 300 < x < 380:  # a deliberate skip
            continue
        d.ellipse([x, 108, x + 16, 132], fill=(196, 170, 120), outline=(120, 100, 70))
    img.save(path)
    return path


def check_vision() -> bool:
    _rule("4. Vision model, weld inspection")
    client, model = nebius.client(), nebius.vision_model()
    img = _test_image(Path("data/captures/preflight_bead.png"))

    schema_hint = (
        'Reply with JSON only: {"verdict":"pass"|"defect",'
        '"defect_type":"none"|"undercut"|"skip"|"porosity"|"spatter"|"burn_through",'
        '"reasoning":"one sentence"}'
    )

    try:
        t0 = time.perf_counter()
        res = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Inspect this weld bead for defects. " + schema_hint,
                        },
                        nebius.image_part(img),
                    ],
                }
            ],
            max_tokens=400,
        )
        dt = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD} vision call failed: {type(exc).__name__}: {exc}")
        return False

    raw = nebius.strip_reasoning(res.choices[0].message.content)
    print(f"{OK} responded in {dt:.0f} ms using {model}")
    print(f"{INFO} {raw[:300]}")
    print(f"{INFO} test image: {img}  (has a deliberate gap, a correct answer names a skip)")
    return True


def main() -> int:
    print("Nebius Token Factory preflight")
    steps = [check_key, check_catalog, check_text, check_vision]
    for step in steps:
        if not step():
            print(f"\nStopped at: {step.__name__}. Fix the above, then rerun.")
            return 1
    print("\nAll four checks passed. The cloud half is wired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
