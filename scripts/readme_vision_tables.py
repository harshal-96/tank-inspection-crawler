"""Write the two vision tables in the README from the studies, so they cannot drift.

    python scripts/readme_vision_tables.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from service import evidence  # noqa: E402

README = Path(__file__).resolve().parent.parent / "README.md"


def field_table() -> str:
    f = evidence.vision_field_study()
    c = f["counts"]
    said = {"judged": "judges it", "declined": "takes it, will not judge a rendering",
            "refused": "refuses any picture", "error": "failed"}
    lines = ["| What happened | Models |", "| --- | --- |"]
    for key in ("refused", "declined", "judged"):
        names = ", ".join(m["raw_model"] for m in f["models"] if m["outcome"] == key)
        lines.append(f"| {said[key]} ({c[key]} of {f['n']}) | {names} |")
    return "\n".join(lines)


def beads_table() -> str:
    v = evidence.vision_study()
    lines = ["| | Beads judged | Named right | Pass/defect right | False alarms | Missed "
             "| Declined | Defect types named | Median time |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for a in v["arms"]:
        name = f"**{a['model']}**" if a["model"] == v["cell_model"] else a["model"]
        lines.append(f"| {name} | {a['judged']} of {a['n']} | {a['named_correct']} | "
                     f"{a['binary_correct']} | {a['false_alarms']} | {a['missed']} | "
                     f"{a['declined']} | {a['defects_named']} | "
                     f"{a['median_latency_ms'] / 1000:.1f} s |")
    return "\n".join(lines)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    for tag, body in (("vision-field", field_table()), ("vision-beads", beads_table())):
        pattern = re.compile(rf"(<!-- {tag}:begin -->\n).*?(<!-- {tag}:end -->)", re.S)
        text, n = pattern.subn(lambda m, b=body: f"{m.group(1)}{b}\n{m.group(2)}", text, count=1)
        if not n:
            raise SystemExit(f"README has no {tag} markers")
    README.write_text(text, encoding="utf-8")
    print("README vision tables written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
