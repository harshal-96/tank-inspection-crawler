"""Read the evaluation runs off disk and serve them as findings, not as files.

Every campaign in this repo leaves a JSONL trail. Those files are the evidence
behind the claims, but a file is not a finding: nobody reads 195 lines of JSON and
comes away knowing whether the small model was right. This module does the
arithmetic once, on the server, so the dashboard renders conclusions instead of
rows and the same numbers appear in the write-up, the API and the charts.

Nothing here computes anything the runs did not already measure. If a file is
missing the study simply reports itself as absent, because a study with no data is
a real answer and an empty chart is not.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Which drift file came from which model. The default run predates the per-model
# naming, so it is spelled out rather than guessed from the filename.
DRIFT_RUNS: dict[str, tuple[str, str]] = {
    "nano": ("drift_eval.jsonl", "Nemotron 3 Nano 30B-A3B"),
    "lightning": ("drift_eval_Nemotron-3_5-Lightning.jsonl", "Nemotron 3.5 Lightning"),
    "super": ("drift_eval_nemotron-3-super-120b-a12b.jsonl", "Nemotron 3 Super 120B-A12B"),
    "ultra": ("drift_eval_Nemotron-3-Ultra-550b-a55b.jsonl", "Nemotron 3 Ultra 550B"),
}

# The model trained on Token Factory and run on the workstation beside the cell,
# and the same base untrained, which is what says whether the training did anything.
DRIFT_EDGE: dict[str, tuple[str, str]] = {
    "edge": ("drift_eval_edge.jsonl", "Qwen3 1.7B, fine-tuned, on the cell's GPU"),
    "edge_base": ("drift_eval_edge_base.jsonl", "Qwen3 1.7B, untrained, on the cell's GPU"),
}

DRIFT_SEVERE: dict[str, tuple[str, str]] = {
    "nano": ("drift_sev2.5_NVIDIA-Nemotron-3-Nano-30B-A3B.jsonl", "Nemotron 3 Nano 30B-A3B"),
    "super": ("drift_sev2.5_nemotron-3-super-120b-a12b.jsonl", "Nemotron 3 Super 120B-A12B"),
}

ARM_LABELS: dict[str, str] = {
    "nano": "Nano on every look",
    "lightning": "Lightning on every look",
    "super": "Super on every look",
    "ultra": "Ultra on every look",
    "routed": "Routed by evidence",
}

DEFECT_CLASSES = ["none", "skip", "undercut", "porosity", "spatter", "burn_through"]


def _read(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / name
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[float]) -> float | None:
    seq = [v for v in values if v is not None]
    return statistics.fmean(seq) if seq else None


def _median(values: Iterable[float]) -> float | None:
    seq = [v for v in values if v is not None]
    return statistics.median(seq) if seq else None


# ------------------------------------------------------------------ study 1: seam

def seam_study() -> dict[str, Any]:
    """Fifty parts, twenty stations each, scored against the seam's true position."""
    rows = _read("seam_campaign.jsonl")
    if not rows:
        return {"available": False}

    means = [r["mean_mm"] for r in rows]
    worst = [r["max_mm"] for r in rows]
    taught = [r["taught_miss_mm"] for r in rows]
    bows = [r["seam_bow_mm"] for r in rows]

    parts = [
        {
            "trial": r["trial"],
            "taught_miss_mm": r["taught_miss_mm"],
            "mean_mm": r["mean_mm"],
            "max_mm": r["max_mm"],
            "bow_mm": r["seam_bow_mm"],
            "stations": r["stations_measured"],
            "asked": r["stations_asked"],
            "refused": r.get("refused_stations", 0),
            "hottest_j_per_mm": r.get("hottest_j_per_mm"),
            "track_ms": r.get("track_ms"),
        }
        for r in rows
    ]

    return {
        "available": True,
        "n_parts": len(rows),
        "stations_per_part": max(r["stations_asked"] for r in rows),
        "stations_measured": sum(r["stations_measured"] for r in rows),
        "taught": {
            "mean_mm": _mean(taught),
            "max_mm": max(taught),
            "min_mm": min(taught),
        },
        "tracked": {
            "mean_mm": _mean(means),
            "median_mm": _median(means),
            "max_mm": max(means),
            "worst_station_mm": max(worst),
            "sub_mm_parts": sum(1 for m in worst if m < 1.0),
        },
        "bow": {"mean_mm": _mean(bows), "max_mm": max(bows)},
        "perception_ms": {
            "median": _median([r.get("track_ms") for r in rows]),
            "max": max((r["track_ms"] for r in rows if r.get("track_ms")), default=None),
        },
        "refused_stations": sum(r.get("refused_stations", 0) for r in rows),
        "parts": parts,
    }


# -------------------------------------------------------------- study 2: locating

def locating_study() -> dict[str, Any]:
    """Blind, one fixed look, and look-then-decide, on the same fifty parts."""
    routed = _read("campaign50_routed.jsonl")
    fixed = _read("campaign50.jsonl")
    if not routed:
        return {"available": False}

    by_trial = {r["trial"]: r for r in fixed}
    parts = []
    for r in routed:
        other = by_trial.get(r["trial"], {})
        parts.append({
            "trial": r["trial"],
            "blind_mm": r["blind_error_mm"],
            "one_look_mm": r["baseline_error_mm"],
            "routed_mm": r["guided_error_mm"],
            "fixed_model_mm": other.get("guided_error_mm"),
            "looks": len(r.get("looks") or []),
        })

    def summarise(key: str) -> dict[str, Any]:
        seq = [p[key] for p in parts if p.get(key) is not None]
        if not seq:
            return {"available": False}
        return {
            "mean_mm": _mean(seq),
            "median_mm": _median(seq),
            "max_mm": max(seq),
            "sub_mm": sum(1 for v in seq if v < 1.0),
            "n": len(seq),
        }

    return {
        "available": True,
        "n_parts": len(parts),
        "blind": summarise("blind_mm"),
        "one_look": summarise("one_look_mm"),
        "routed": summarise("routed_mm"),
        "fixed_model": summarise("fixed_model_mm"),
        "parts": parts,
    }


# ------------------------------------------------------------- study 3: the router

def router_study() -> dict[str, Any]:
    """Was the look-again call right, judged against what a second look actually did."""
    rows = _read("router_truth_eval.jsonl")
    if not rows:
        return {"available": False}

    arms: dict[str, dict[str, Any]] = {}
    for row in rows:
        arm = arms.setdefault(row["arm"], {
            "arm": row["arm"],
            "label": ARM_LABELS.get(row["arm"], row["arm"]),
            "n": 0, "correct": 0, "said_look": 0, "should_look": 0,
            "latency": [], "tokens": [], "gain_mm": 0.0, "cost_mm": 0.0,
        })
        arm["n"] += 1
        arm["correct"] += 1 if row["correct"] else 0
        arm["said_look"] += 1 if row["said_look_again"] else 0
        arm["should_look"] += 1 if row["should_look_again"] else 0
        arm["latency"].append(row.get("latency_ms"))
        arm["tokens"].append(row.get("tokens"))
        if row["said_look_again"]:
            arm["gain_mm"] += row.get("gain_mm") or 0.0
            arm["cost_mm"] += row.get("cost_mm") or 0.0

    summary = []
    for arm in arms.values():
        looks = arm["n"] + arm["said_look"]
        summary.append({
            "arm": arm["arm"],
            "label": arm["label"],
            "n": arm["n"],
            "correct": arm["correct"],
            "accuracy": arm["correct"] / arm["n"],
            "said_look": arm["said_look"],
            "should_look": arm["should_look"],
            "looks_per_decision": looks / arm["n"],
            "median_latency_ms": _median(arm["latency"]),
            "total_latency_s": sum(v for v in arm["latency"] if v) / 1000.0,
            "median_tokens": _median(arm["tokens"]),
            "gain_mm": arm["gain_mm"],
            "cost_mm": arm["cost_mm"],
        })
    order = ["routed", "nano", "lightning", "super", "ultra"]
    summary.sort(key=lambda a: order.index(a["arm"]) if a["arm"] in order else 99)

    # A per-case grid, so the disagreements are visible rather than averaged away.
    cases = sorted({(r["trial"], r["should_look_again"]) for r in rows})
    grid = [{"trial": t, "should_look": s, "calls": {}} for t, s in cases]
    index = {(g["trial"], g["should_look"]): g for g in grid}
    seen: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["trial"], row["should_look_again"])
        cell = index.get(key)
        if cell is None:
            continue
        slot = seen.get((row["arm"], row["trial"]), 0)
        seen[(row["arm"], row["trial"])] = slot + 1
        cell["calls"].setdefault(row["arm"], []).append({
            "said_look": row["said_look_again"],
            "correct": row["correct"],
        })

    return {"available": True, "arms": summary, "grid": grid}


# ---------------------------------------------------------- study 4: drift, by ear

def drift_study() -> dict[str, Any]:
    """Given a run of parts, name the cause: the press, the fixture, or nothing."""
    def load(spec: dict[str, tuple[str, str]]) -> list[dict[str, Any]]:
        out = []
        for tier, (name, label) in spec.items():
            rows = _read(name)
            if not rows:
                continue
            answers = {r["said"] for r in rows}
            out.append({
                "tier": tier,
                "model": label,
                "n": len(rows),
                "correct": sum(1 for r in rows if r["correct"]),
                "accuracy": sum(1 for r in rows if r["correct"]) / len(rows),
                "median_latency_ms": _median([r.get("latency_ms") for r in rows]),
                "degenerate": len(answers) == 1,
                "always_said": next(iter(answers)) if len(answers) == 1 else None,
                "cases": [
                    {
                        "scenario": r["scenario"],
                        "seed": r["seed"],
                        "said": r["said"],
                        "correct": r["correct"],
                        "confidence": r.get("confidence"),
                        "tilt_slope": r.get("tilt_slope_mm_per_part"),
                        "shift_slope": r.get("shift_slope_mm_per_part"),
                        "evidence": r.get("evidence"),
                    }
                    for r in rows
                ],
            })
        return out

    normal = load(DRIFT_RUNS)
    severe = load(DRIFT_SEVERE)
    edge = load(DRIFT_EDGE)
    if not normal:
        return {"available": False}
    scenarios = sorted({c["scenario"] for run in normal for c in run["cases"]})
    seeds = sorted({c["seed"] for run in normal for c in run["cases"]})
    return {
        "available": True,
        "scenarios": scenarios,
        "seeds": seeds,
        "runs": normal,
        "severe": severe,
        "edge": edge,
    }


# ------------------------------------------------------- study 5: the post-weld eye

def inspection_study() -> dict[str, Any]:
    """Named defects scored against the defect that was injected on purpose."""
    rows = _read("inspections.jsonl")
    if not rows:
        return {"available": False}

    labels = [c for c in DEFECT_CLASSES
              if any(r["injected"] == c or r["called"] == c for r in rows)]
    matrix = [[0] * len(labels) for _ in labels]
    for r in rows:
        if r["injected"] in labels and r["called"] in labels:
            matrix[labels.index(r["injected"])][labels.index(r["called"])] += 1

    clean = [r for r in rows if r["injected"] == "none"]
    faulty = [r for r in rows if r["injected"] != "none"]
    false_alarm = sum(1 for r in clean if r["called"] != "none")
    missed = sum(1 for r in faulty if r["called"] == "none")
    named = sum(1 for r in rows if r["correct"])
    binary = sum(1 for r in rows if (r["injected"] == "none") == (r["called"] == "none"))

    return {
        "available": True,
        "n": len(rows),
        "model": rows[0].get("model"),
        "labels": labels,
        "matrix": matrix,
        "named_correct": named,
        "named_accuracy": named / len(rows),
        "binary_correct": binary,
        "binary_accuracy": binary / len(rows),
        "clean": len(clean),
        "faulty": len(faulty),
        "false_alarms": false_alarm,
        "missed": missed,
        "median_latency_ms": _median([r.get("latency_ms") for r in rows]),
        "cases": [
            {
                "trial": r["trial"],
                "injected": r["injected"],
                "called": r["called"],
                "correct": r["correct"],
                "severity": r.get("severity"),
                "latency_ms": r.get("latency_ms"),
                "reasoning": r.get("reasoning"),
            }
            for r in rows
        ],
    }


VISION_RUNS: dict[str, tuple[str, str]] = {
    "gemma": ("inspections.jsonl", "google/gemma-3-27b-it"),
    "minicpm": ("inspections_minicpm.jsonl", "openbmb/MiniCPM-V-4_5"),
}


def _score_inspection(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    clean = [r for r in rows if r["injected"] == "none"]
    faulty = [r for r in rows if r["injected"] != "none"]
    called_any = {r["called"] for r in rows}
    return {
        "model": label,
        "n": len(rows),
        "named_correct": sum(1 for r in rows if r["correct"]),
        "binary_correct": sum(1 for r in rows
                              if (r["injected"] == "none") == (r["called"] == "none")),
        "false_alarms": sum(1 for r in clean if r["called"] != "none"),
        "missed": sum(1 for r in faulty if r["called"] == "none"),
        "clean": len(clean),
        "faulty": len(faulty),
        "defects_named": len(called_any - {"none"}),
        "never_says_defect": called_any == {"none"},
        "median_latency_ms": _median([r.get("latency_ms") for r in rows]),
    }


def vision_study() -> dict[str, Any]:
    """Both vision models the account serves, on the same fifty beads.

    Worth its own study because the two scores are nearly identical and mean
    opposite things. One of them reaches that score by naming defects and getting
    some of them wrong; the other reaches it by never once using the word. A
    single accuracy figure cannot tell those apart, which is the argument for
    scoring the two kinds of mistake separately everywhere else on this dashboard.
    """
    arms = []
    for _, (name, label) in VISION_RUNS.items():
        rows = _read(name)
        if rows:
            arms.append(_score_inspection(rows, label))
    if not arms:
        return {"available": False}
    return {"available": True, "arms": arms}


def all_studies() -> dict[str, Any]:
    return {
        "seam": seam_study(),
        "locating": locating_study(),
        "router": router_study(),
        "drift": drift_study(),
        "inspection": inspection_study(),
        "vision": vision_study(),
    }
