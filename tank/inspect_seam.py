"""Judge one frame of a tank seam.

This is the welding cell's inspector doing the same job on a different wall,
and it is worth being exact about what is shared and what is not.

Shared: the call policy in brain/nebius.py, so an API option a model rejects is
dropped and retried and a reply that spent its budget thinking is asked again
with room. The JSON contract. The refusal to let a parse failure become a
verdict. The rule that a declined frame is scored in its own column rather
than as right or wrong.

Not shared: the words. The cell inspects fillet welds on folded trays and its
vocabulary is skip, undercut, porosity, spatter, burn-through. A tank shell is
judged on cracks and lack of fusion as well, and has corrosion on the plate
beside the weld that a production cell never sees. Reusing the cell's five
words would mean scoring a crack as whatever it happened to be called, and
extending the cell's list would mean every number already measured with it was
measured under a different prompt. So the tank gets its own, and the two sets
of results stay comparable to themselves.

The framing lesson carries over intact. The first version of the cell's prompt
called a render a photograph and offered "uncertain" as a way out, and the
better models took it. This one says what the picture is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from brain import nebius

# What a tank seam is judged on. The order is kept because a single headline
# finding is still useful on a work order, but it is no longer what the
# inspector is scored on: see the note in score_survey.py about why ranking a
# corroded shell below a weld-toe groove is the wrong way round for a tank that
# is already in service.
SEAM_DEFECTS = ("crack", "burn_through", "lack_of_fusion", "undercut",
                "porosity", "corrosion", "none")
SEVERITY = {kind: index for index, kind in enumerate(SEAM_DEFECTS)}

# The same order, in words, for the prompt. Written out from the tuple above
# rather than typed a second time, because a precedence the scorer applies and
# the inspector has not been given is not a test of the inspector.
PRECEDENCE = ", ".join(kind for kind in SEAM_DEFECTS if kind != "none")

SYSTEM_PROMPT = """You are inspecting one length of weld seam inside a steel \
storage tank, as a robot crawling the shell sees it.

The picture is a rendering from the simulator this crawler is tested against, \
not a photograph. The seam is a raised bead of weld metal running across the \
frame, with a darker heat-tinted band on the plate either side of it. Judge the \
rendering exactly as you would a photograph. That the image is synthetic is not \
a reason to answer uncertain.

What each fault looks like.
- crack: a fine dark line in the bead or running across it, much longer than it \
is wide and narrower than a pit.
- burn_through: one dark hole where the weld metal is missing, about as wide as \
the bead or wider, with the cavity behind showing through.
- lack_of_fusion: the bead stops and starts again, leaving a length of bare \
joint with no weld metal on it at all.
- undercut: a dark groove melted into the plate along the toe of the bead. It \
runs beside the bead and never through it.
- porosity: several small dark rounded pits in the face of the bead itself, \
each narrower than the bead.
- corrosion: a rust-coloured pitted patch on the plate away from the weld. The \
bead itself is sound.

The seam is a pass ONLY if the bead runs continuously across the frame and \
shows none of the above.

Look at three places before you answer, every time: the face of the bead, the \
line where the bead meets the plate on both sides of it, and the plate away \
from the weld.

Examine three regions separately and report on each of them by name, every \
time, even when there is nothing to say about one: the face of the bead, the \
line where the bead meets the plate on both sides of it, and the plate away \
from the weld. A groove at the toe is easy to walk past when the plate beside \
it is badly corroded, so the toe is looked at on its own.

Then list EVERY fault you can see, not only the most serious one. A frame can \
carry a fault in the weld and corrosion on the plate at the same time, and both \
go in the report, because they are two different jobs for two different people.

Then name the one that would stop the job first, out of the ones you listed. \
Where more than one is in the frame, that is the one coming first in this \
order:
""" + PRECEDENCE + """

Answer "uncertain" only if no bead is in the frame at all.

Reply with JSON only, no prose and no code fence:
{"verdict": "pass" | "defect" | "uncertain",
 "bead_face": "what you see on the face of the bead itself",
 "bead_toes": "what you see along both lines where the bead meets the plate",
 "plate": "what you see on the plate away from the weld",
 "faults": ["every fault visible, by name, from the list above; empty if none"],
 "worst": "none" | "crack" | "burn_through" | "lack_of_fusion" | "undercut" | \
"porosity" | "corrosion",
 "severity": 0.0 to 1.0,
 "where": "left, middle or right of the frame",
 "reasoning": "one sentence citing what you saw"}"""

USER_PROMPT = "Inspect this length of seam."


@dataclass
class SeamVerdict:
    verdict: str
    defect_type: str
    faults: list = field(default_factory=list)
    severity: Optional[float] = None
    where: str = ""
    reasoning: str = ""
    model: Optional[str] = None
    latency_ms: float = 0.0
    retried_with_room: bool = False
    parse_error: Optional[str] = None
    raw: str = ""

    def as_row(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "called": self.defect_type,
                "faults": list(self.faults), "raw": self.raw[:400],
                "severity": self.severity, "where": self.where,
                "reasoning": self.reasoning, "model": self.model,
                "latency_ms": round(self.latency_ms), "parse_error": self.parse_error,
                "retried_with_room": self.retried_with_room}


@dataclass
class _Coerced:
    verdict: str
    defect: str
    severity: Optional[float]
    faults: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def _coerce(data: dict[str, Any]) -> _Coerced:
    """Keep the two fields consistent without inventing a verdict.

    A model that says defect and then names nothing has not judged the frame,
    so it is recorded as uncertain rather than quietly turned into a pass.
    """
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "defect", "uncertain"):
        verdict = "uncertain"
    # Anything outside the vocabulary is dropped rather than guessed at, and
    # the order the model listed them in is not trusted: they are ranked here.
    named = data.get("faults") or []
    if isinstance(named, str):
        named = [named]
    faults = []
    for item in named:
        kind = str(item).strip().lower()
        if kind in SEAM_DEFECTS and kind != "none" and kind not in faults:
            faults.append(kind)
    faults.sort(key=lambda k: SEVERITY[k])
    defect = str(data.get("worst", data.get("defect_type", "none"))).strip().lower()
    if defect not in SEAM_DEFECTS:
        defect = "none"
    # A headline that contradicts the list is resolved in favour of the list,
    # since that is the part being measured.
    if faults and defect not in faults:
        defect = faults[0]
    if verdict == "pass":
        defect, faults = "none", []
    elif verdict == "defect" and defect == "none":
        verdict = "uncertain"
    severity = data.get("severity")
    try:
        severity = min(1.0, max(0.0, float(severity))) if severity is not None else None
    except (TypeError, ValueError):
        severity = None
    return _Coerced(verdict, defect, severity, faults,
                    {"where": str(data.get("where") or "").strip()[:40],
                     "reasoning": str(data.get("reasoning") or "").strip()})


def inspect_seam(image_path: str | Path, model: Optional[str] = None,
                 detail: str = "high", max_tokens: int = 700,
                 api: Optional[Any] = None) -> SeamVerdict:
    """One frame in, one verdict out. Raises only on transport failure."""
    model = model or nebius.vision_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": USER_PROMPT},
            nebius.image_part(image_path, detail=detail),
        ]},
    ]
    started = time.perf_counter()
    reply = nebius.chat(messages, model=model, max_tokens=max_tokens,
                        parses=nebius.extract_json, api=api)
    elapsed = (time.perf_counter() - started) * 1000

    try:
        data = nebius.extract_json(reply.raw)
    except Exception as exc:  # noqa: BLE001  any unreadable reply
        return SeamVerdict("uncertain", "none", model=model, latency_ms=elapsed,
                           retried_with_room=reply.retried_with_room,
                           parse_error=f"{type(exc).__name__}: {exc}",
                           reasoning="Model reply could not be read.", raw=reply.raw)

    got = _coerce(data)
    return SeamVerdict(got.verdict, got.defect, got.faults, got.severity,
                       got.extras["where"], got.extras["reasoning"],
                       model=model, latency_ms=elapsed,
                       retried_with_room=reply.retried_with_room, raw=reply.raw)


def worst(kinds) -> str:
    """The fault that would stop the job first, out of everything in one frame."""
    ranked = sorted((k for k in kinds if k in SEVERITY), key=lambda k: SEVERITY[k])
    return ranked[0] if ranked else "none"
