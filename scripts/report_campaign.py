"""Turn campaign results into the table that goes in the README.

Generated rather than transcribed, so a published number cannot drift from the
run that produced it.

    python scripts/report_campaign.py \\
        "camera well placed=data/campaign_good_view.jsonl" \\
        "camera poorly placed=data/campaign_poor_view.jsonl" \\
        --out docs/results.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Stats:
    n: int
    attempted: int
    mean: Optional[float] = None
    median: Optional[float] = None
    p95: Optional[float] = None
    worst: Optional[float] = None
    sub_mm: int = 0

    @property
    def failed(self) -> int:
        return self.attempted - self.n

    def cell(self, value: Optional[float]) -> str:
        return "n/a" if value is None else f"{value:.2f}"

    def row(self, label: str) -> str:
        """A row that cannot flatter itself.

        How many parts an arm actually managed to measure belongs in the table, not
        in a footnote: an arm that fails on eleven parts and reports a tidy number
        for the twelfth has not done better, it has done worse.
        """
        measured = f"{self.n} of {self.attempted}"
        if self.n == 0:
            return f"| {label} | {measured} | no measurement | | | | 0 |"
        return (
            f"| {label} | {measured} | {self.cell(self.mean)} | "
            f"{self.cell(self.median)} | {self.cell(self.p95)} | "
            f"{self.cell(self.worst)} | {self.sub_mm} of {self.attempted} |"
        )


def summarise(values: list[Optional[float]]) -> Stats:
    good = [v for v in values if v is not None]
    if not good:
        return Stats(n=0, attempted=len(values))
    ordered = sorted(good)
    index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered) + 0.5) - 1))
    return Stats(
        n=len(good),
        attempted=len(values),
        mean=statistics.fmean(good),
        median=statistics.median(good),
        p95=ordered[index],
        worst=max(good),
        sub_mm=sum(1 for v in good if v < 1.0),
    )


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "campaigns",
        nargs="+",
        help='One or more "label=path.jsonl" pairs, one per fixed-camera placement.',
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sources = []
    for entry in args.campaigns:
        if "=" not in entry:
            raise SystemExit(f"expected label=path, got {entry!r}")
        label, path = entry.split("=", 1)
        sources.append((label.strip(), Path(path.strip())))

    first = load(sources[0][1])
    lines = [
        "# Measured results",
        "",
        "Joint position error against the true joint, in millimetres. Parts arrive",
        "under physics with randomised offset. Every arm reads the same renders of the",
        "same parts, and turning the part about the joint axis leaves the joint itself",
        "where it is, so the target never moves between views.",
        "",
        f"Parts per campaign: {len(first)}.",
        "",
        "| | Measured | Mean | Median | p95 | Worst | Sub-millimetre |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        summarise([r["blind_error_mm"] for r in first]).row("No correction"),
    ]

    guided_rows = []
    for label, path in sources:
        rows = load(path)
        lines.append(
            summarise([r["baseline_error_mm"] for r in rows]).row(f"One fixed view, {label}")
        )
        guided_rows.append((label, summarise([r["guided_error_mm"] for r in rows]), rows))

    for label, stats, _ in guided_rows:
        lines.append(stats.row(f"Look, judge, look again, starting {label}"))

    lines += ["", "## What the loop costs", ""]
    for label, _, rows in guided_rows:
        looks = [len(r["looks"]) for r in rows]
        calls = sum(len(r["decisions"]) for r in rows)
        latency = [
            d["latency_ms"] for r in rows for d in r["decisions"] if d.get("latency_ms")
        ]
        lines.append(
            f"- Starting {label}: {statistics.fmean(looks):.2f} looks per part on average, "
            f"at most {max(looks)}. {calls} model calls over {len(rows)} parts"
            + (
                f", median {statistics.median(latency):.0f} ms each."
                if latency
                else "."
            )
        )

    report = "\n".join(lines) + "\n"
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
