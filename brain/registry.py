"""One name per model, decided here rather than at each call site.

The same model was appearing under five different labels across the studies:
`super`, `Nemotron 3 Super 120B`, `Nemotron 3 Super 120B-A12B`, the raw
`nvidia/nemotron-3-super-120b-a12b`, and `Super 120B, boundary withheld`. Each
script had picked its own, so a reader moving between two pages could not tell
whether they were looking at one model or two, and counting the models in the
project gave a different answer depending on which page you counted from.

So every model this project calls is listed once, with the identifier the API
wants, the name a reader sees, and what it is here to do. `display()` takes any
of the labels that have been used historically and returns the canonical entry,
so the old result files keep working without being rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Model:
    """A model this project calls, and the one name it goes by."""

    key: str                    # short, stable, used in data files
    api_id: str                 # what Token Factory wants
    name: str                   # what a reader sees
    params: str                 # as the publisher states it
    role: str                   # why this project calls it
    vendor: str
    served_by: str = "Nebius Token Factory"
    aliases: tuple[str, ...] = field(default_factory=tuple)


# The four sizes scored on the per-part question, smallest first. The smallest runs.
SIZES = (
    Model("nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
          "Nemotron 3 Nano 30B", "30B-A3B",
          "Every per-part decision: is this measurement good enough to weld on. "
          "Also measured as the tank's probe planner: never ahead of a coverage rule, "
          "and level with it when the rule is shown only what the model sees",
          "NVIDIA",
          aliases=("Nemotron 3 Nano 30B-A3B", "NVIDIA-Nemotron-3-Nano-30B-A3B",
                   "nemotron-3-nano-30b-a3b")),
    Model("lightning", "nvidia/Nemotron-3_5-Lightning",
          "Nemotron 3.5 Lightning", "undisclosed",
          "Drives the supervisor agent and writes the corrective action briefs; "
          "scored on the per-part decision, not routed to",
          "NVIDIA",
          aliases=("Nemotron-3_5-Lightning", "Nemotron 3.5 Lightning")),
    Model("super", "nvidia/nemotron-3-super-120b-a12b",
          "Nemotron 3 Super 120B", "120B-A12B",
          # Asked the same boundary-stated question twice: 44 in data/hard64_a.jsonl
          # and 46 as the "cloud" arm of data/hard64_c.jsonl, which
          # service/evidence.hard_study() pools as 90 of 128. The diagnosis default
          # is DeepSeek V4.1 Flash (brain/nebius.py), so Super is measured here,
          # not called in the loop.
          "Scored on the drift question (44 and 46 of 64 in two runs), not deployed",
          "NVIDIA",
          aliases=("Nemotron 3 Super 120B-A12B", "nemotron-3-super-120b-a12b",
                   "nemotron-3-super-120", "Super 120B, boundary withheld")),
    Model("ultra", "nvidia/Nemotron-3-Ultra-550b-a55b",
          "Nemotron 3 Ultra 550B", "550B-A55B",
          "Scored on the drift question as the ceiling, not routed to",
          "NVIDIA",
          aliases=("Nemotron 3 Ultra 550B-A55B", "Nemotron-3-Ultra-550",
                   "Nemotron-3-Ultra-550b-a55b")),
)

# Everything else the project calls, grouped by the job it does.
VISION = (
    Model("minimax", "MiniMaxAI/MiniMax-M3", "MiniMax-M3", "undisclosed",
          "Names the defect best, 117 of 150, but ships 20 bad parts to DeepSeek's 14",
          "MiniMax", aliases=("MiniMax-M3",)),
    Model("gemma", "google/gemma-3-27b-it", "Gemma 3 27B", "27B",
          "The first inspector: 131 of 150 pass or defect, 18 missed, 1.9 s", "Google",
          aliases=("gemma-3-27b-it",)),
    Model("minicpm", "openbmb/MiniCPM-V-4_5", "MiniCPM-V 4.5", "8B",
          "Scored on the same beads as a second opinion", "OpenBMB",
          aliases=("MiniCPM-V-4_5",)),
)

EDGE = (
    Model("edge", "Qwen/Qwen3-1.7B", "Qwen3 1.7B, fine-tuned", "1.7B",
          "Runs on the workstation beside the cell, with no network",
          "Alibaba", served_by="trained on Token Factory, run locally",
          aliases=("Qwen3-1.7B + LoRA, on the cell", "edge", "edge4",
                   "Qwen3 1.7B, fine-tuned, on the cell's GPU",
                   "Qwen3 1.7B, fine-tuned on four causes, on the cell's GPU")),
    Model("edge_base", "Qwen/Qwen3-1.7B", "Qwen3 1.7B, untrained", "1.7B",
          "The same model before fine-tuning, as the control",
          "Alibaba", served_by="run locally",
          aliases=("edge_base", "edge4_base",
                   "Qwen3 1.7B, untrained, on the cell's GPU")),
)

# Run only on the boundary test, to see where the whole field sits.
FIELD = (
    Model("deepseek_pro", "deepseek-ai/DeepSeek-V4-Pro", "DeepSeek V4 Pro",
          "undisclosed", "Boundary test only", "DeepSeek",
          aliases=("DeepSeek-V4-Pro",)),
    Model("deepseek_flash", "deepseek-ai/DeepSeek-V4.1-Flash", "DeepSeek V4.1 Flash",
          "undisclosed",
          "Judges the bead (136 of 150 pass or defect, 14 missed, none scrapped) and names "
          "the machine that slipped once a shift (45 of 64 on the boundary test)",
          "DeepSeek", aliases=("DeepSeek-V4.1-Flash",)),
    Model("qwen397", "Qwen/Qwen3.5-397B-A17B", "Qwen3.5 397B", "397B-A17B",
          "Boundary test only", "Alibaba",
          aliases=("Qwen3.5-397B-A17B",)),
    Model("gptoss", "openai/gpt-oss-120b", "gpt-oss 120B", "120B",
          "Boundary test only", "OpenAI",
          aliases=("gpt-oss-120b",)),
)

# Every other model on the account that will take an image at all, found by
# putting all of them through the inspection path (scripts/probe_vision_field.py)
# and then scored on the same fifty beads as Gemma. The earlier claim that only
# two models could see was measured on five candidates and stated as if it
# covered the account. The roles quote the corrected prompt, which tells the
# model the picture is a rendering; under the first prompt these models
# mostly declined, and that was the prompt's fault.
VISION_FIELD = (
    Model("glm_flash", "zai-org/GLM-5.3-Flash", "GLM-5.3 Flash", "undisclosed",
          "33 of 44 named right, still declines some beads, 8.8 s", "Zhipu",
          aliases=("GLM-5.3-Flash",)),
    Model("kimi", "moonshotai/Kimi-K2.6", "Kimi K2.6", "undisclosed",
          "33 of 42 named right, still declines some beads, 11.8 s", "Moonshot",
          aliases=("Kimi-K2.6",)),
)

ALL = SIZES + VISION + EDGE + FIELD + VISION_FIELD

_BY_LABEL: dict[str, Model] = {}
for _m in ALL:
    for _label in (_m.key, _m.api_id, _m.name, *_m.aliases):
        _BY_LABEL[_label.strip().lower()] = _m


def find(label: object) -> Optional[Model]:
    """The canonical model behind any label this project has ever used for it."""
    if label is None:
        return None
    text = str(label).strip()
    hit = _BY_LABEL.get(text.lower())
    if hit:
        return hit
    # Result files sometimes carry a truncated arm name, so fall back to the
    # longest registered label that the text starts with.
    starts = [lbl for lbl in _BY_LABEL if text.lower().startswith(lbl) and len(lbl) > 6]
    return _BY_LABEL[max(starts, key=len)] if starts else None


def display(label: object, fallback: Optional[str] = None) -> str:
    """The reader-facing name, or the label unchanged when it is not a model."""
    hit = find(label)
    if hit:
        return hit.name
    return fallback if fallback is not None else str(label)


def summary() -> list[dict]:
    """Every model, in the order a reader should meet them."""
    groups = [("Scored per part; the smallest one runs", SIZES), ("Reads the weld bead", VISION),
              ("Runs on the cell itself", EDGE), ("Scored on the boundary test only", FIELD),
              ("Will take an image; scored on the beads", VISION_FIELD)]
    out = []
    for group, models in groups:
        for m in models:
            out.append({"group": group, "key": m.key, "name": m.name,
                        "api_id": m.api_id, "params": m.params, "role": m.role,
                        "vendor": m.vendor, "served_by": m.served_by})
    return out
