"""The cell supervisor: an NVIDIA Nemotron model with tools over the production database.

It reads the measurement stream, explains what it sees to an operator, and records
recommendations. It does not command the cell. Every decision lands in the database
awaiting acknowledgement, which is how a supervisor on a real line works.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from sqlmodel import Session

from brain import nebius
from brain.tools import TOOL_SCHEMAS, dispatch

MAX_TOOL_ROUNDS = 6
MAX_TOKENS = 1200

# Agent replies are shown on the operator dashboard and may be filmed, so they count
# as public text. Models reach for typographic punctuation; this maps it back to ASCII.
_DASHES = re.compile(r"\s*[—–‒]\s*")
_SIMPLE = {
    "‑": "-",    # non-breaking hyphen
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    " ": " ",    # non-breaking space
}


def plain_text(text: str) -> str:
    """Normalise model punctuation to ASCII.

    Em and en dashes become commas, which reads correctly in the mid-sentence
    position models use them in.
    """
    text = _DASHES.sub(", ", text)
    for fancy, plain in _SIMPLE.items():
        text = text.replace(fancy, plain)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"[ 	]{2,}", " ", text)
    return re.sub(r" +([,.;:!?])", r"\1", text).strip()


SYSTEM_PROMPT = """You are the supervisor of a robotic welding cell that corrects \
itself. Your audience is the operator on the floor, not an engineer at a desk.

How the cell works. Sheet metal parts arrive and are clamped in a fixture. Part \
variation, meaning camber, springback and fit-up, moves the real joint away from the \
taught robot path. Before welding, a camera photographs the part and a detector finds \
where the joint actually is, so the robot retargets. After welding, a second camera \
photographs the bead and a vision model judges it.

What the numbers mean.
- Correction error is the distance in millimetres between where the detector said the \
joint was and where it really was. Under 1 mm is good. Over 3 mm is a problem.
- Lateral drift compares the recent parts against earlier ones. Random scatter is \
normal. A steady one-directional shift is mechanical: a locator wearing, a fixture \
settling, or a clamp losing pressure. That is worth flagging even when every \
individual part still passes, because it will not stop on its own.
- Inspection accuracy is scored against defects that were deliberately injected, so a \
missed defect and a false alarm are both known and countable.

How to work.
- Call a tool for every number. Never estimate, never recall a figure from earlier in \
the conversation, and never do arithmetic yourself. If a tool did not give you a \
number, say you do not have it.
- Prefer one good tool call over several. Start with get_cell_status for general \
questions.
- If you recommend anything, you MUST also log it with record_decision in the same \
reply. Saying it in prose is not enough: only a recorded decision reaches the \
operator's queue, so an unlogged recommendation is lost. Gather evidence first, then \
record, then explain. Use hold_line to stop the line, recalibrate for a drift or \
accuracy problem, rework or accept for a single part. Record at most ONE decision per \
reply, the single most important one. Once it is recorded, move straight to your \
answer.
- Every figure in a rationale must come from a tool result in this reply. Do not carry \
a number over from an earlier turn: the line has moved on since then.
- A recorded decision is still a recommendation, not a command. It does not move the \
cell. Tell the operator it is awaiting their acknowledgement.
- Answer in plain language, a few sentences. Give the number, then what it means, then \
what you suggest. No headings, no bullet lists unless asked, no restating the question.
- If the cell is running well, say so briefly rather than manufacturing concern.
- Write with plain ASCII punctuation only. Never use em dashes or en dashes. Use
commas, colons or full stops instead."""

REPORT_PROMPT = """Write the end-of-shift report for this cell.

Gather what you need with the tools first. Then write it as short prose in Markdown \
under 250 words, with these sections as level-3 headings: Output, Accuracy, Drift, \
Actions.

State numbers plainly with units. If something needs a human, say so and say why. Do \
not invent anything the tools did not give you, and do not pad."""


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "result": self.result}


@dataclass
class AgentReply:
    reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    rounds: int = 0
    truncated: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "tool_calls": [c.as_dict() for c in self.tool_calls],
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


def _parse_arguments(raw: Optional[str]) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string and are occasionally malformed."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _run(
    messages: list[dict[str, Any]],
    session: Session,
    model: str,
    max_rounds: int = MAX_TOOL_ROUNDS,
    max_tokens: int = MAX_TOKENS,
) -> AgentReply:
    """Drive the tool-calling loop until the model answers in prose."""
    client = nebius.client()
    started = time.perf_counter()
    reply = AgentReply(reply="", model=model)
    recorded: list[int] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}

    for round_index in range(1, max_rounds + 1):
        reply.rounds = round_index
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            # Force a lookup on the opening round. With conversation history in
            # context the model will otherwise reuse a figure it quoted earlier,
            # which goes stale the moment another part is measured. Grounding the
            # first round costs one call and removes a whole class of wrong answer.
            tool_choice="required" if round_index == 1 else "auto",
            max_tokens=max_tokens,
        )
        if response.usage:
            reply.prompt_tokens += response.usage.prompt_tokens
            reply.completion_tokens += response.usage.completion_tokens

        message = response.choices[0].message
        calls = message.tool_calls or []

        if not calls:
            reply.reply = plain_text(nebius.strip_reasoning(message.content))
            break

        messages.append(message.model_dump(exclude_none=True))
        for call in calls:
            name = call.function.name
            arguments = _parse_arguments(call.function.arguments)

            # One decision per reply. Without this the model logs a decision, does
            # not treat it as done, and logs it again, producing a queue of
            # contradictory recommendations from a single question.
            if name == "record_decision" and recorded:
                result: dict[str, Any] = {
                    "error": (
                        f"A decision was already recorded in this reply (id={recorded[0]}). "
                        "Record at most one. Summarise it for the operator now."
                    )
                }
            else:
                # Read tools are memoised per reply. The model otherwise re-queries
                # the same figures every round, which costs a database hit and a
                # round trip without changing the answer.
                key = (name, json.dumps(arguments, sort_keys=True, default=str))
                if name != "record_decision" and key in seen:
                    result = dict(seen[key])
                    result["note"] = (
                        "You already called this in this reply. Use the values above "
                        "and write your answer now."
                    )
                else:
                    result = dispatch(name, arguments, session, model=model)
                    if name == "record_decision" and result.get("recorded"):
                        recorded.append(result["decision_id"])
                    else:
                        seen[key] = result

            reply.tool_calls.append(ToolCallRecord(name, arguments, result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )
    else:
        # Ran out of rounds while still calling tools.
        reply.truncated = True
        reply.reply = (
            "I gathered the data but ran out of steps before summarising it. "
            "The tool results are attached."
        )

    if not reply.reply:
        reply.reply = "No answer was produced."

    reply.latency_ms = (time.perf_counter() - started) * 1000
    return reply


def ask(
    message: str,
    session: Session,
    history: Optional[list[dict[str, Any]]] = None,
    model: Optional[str] = None,
) -> AgentReply:
    """Answer one operator question, calling tools as needed."""
    model = model or nebius.text_model()
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})
    return _run(messages, session, model)


def shift_report(session: Session, model: Optional[str] = None) -> AgentReply:
    """Generate an end-of-shift report from the data on record."""
    model = model or nebius.text_model()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REPORT_PROMPT},
    ]
    return _run(messages, session, model, max_tokens=1600)
