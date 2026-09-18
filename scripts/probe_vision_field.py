"""Which models on this account can inspect a weld bead at all?

The inspection page said the account serves two models that can see. That was
measured on five candidates and stated as if it covered the account. It did not:
a later probe found MiniMax-M3 and GLM-5.3-Flash will take an image and name a
defect, DeepSeek V4.1 Flash accepts one and answers nothing under the budget
this project sends, and the two Kimi models see the picture well enough to say
it is a rendering and decline to judge it.

So every model the key can call is put through the real inspection path, on one
clean bead and one with a defect, and what comes back is recorded: a verdict,
a parse failure, or the platform refusing the image. A claim about which models
can see should rest on this file and not on whoever last remembered.

    python scripts/probe_vision_field.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brain import nebius  # noqa: E402
from brain.inspector import inspect_weld  # noqa: E402

BEADS = [
    ("clean", "data/demo/beads/rgb_0009.png", "none"),
    ("defect", "data/demo/beads/rgb_0039.png", "burn_through"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/vision_field.jsonl")
    args = parser.parse_args()

    models = [m for m in nebius.available_models() if "Embedding" not in m]
    print(f"{len(models)} models, two beads each\n")
    print(f"{'model':<40} {'clean bead':<22} {'defect bead':<22} {'s':>6}")
    print("-" * 96)

    records = []
    for model in models:
        row = {"model": model}
        started = time.perf_counter()
        cells = []
        for label, path, injected in BEADS:
            try:
                r = inspect_weld(path, model=model)
                outcome = ("unparsed" if r.parse_error else r.defect_type.value)
                row[label] = {
                    "verdict": r.verdict.value, "called": r.defect_type.value,
                    "injected": injected, "parse_error": r.parse_error,
                    "retried_with_room": bool(r.extra.get("retried_with_room")),
                    "latency_ms": round(r.latency_ms or 0),
                }
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                refused = "image" in msg.lower() or "support" in msg.lower()
                outcome = "no image input" if refused else f"error: {type(exc).__name__}"
                row[label] = {"error": outcome, "detail": msg[:160]}
            cells.append(outcome)
        row["seconds"] = round(time.perf_counter() - started, 1)
        records.append(row)
        print(f"{model:<40} {cells[0]:<22} {cells[1]:<22} {row['seconds']:>6}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
