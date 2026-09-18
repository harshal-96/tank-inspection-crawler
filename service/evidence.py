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
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from brain import inspector, nebius, registry
from cell.limits import WELDABLE_MM

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

# The same models on the harder question: four causes rather than three, with the
# reference channel that separates a drifting camera from a drifting fixture.
DRIFT_EDGE_FOUR: dict[str, tuple[str, str]] = {
    "edge4": ("drift_eval_edge4.jsonl",
              "Qwen3 1.7B, fine-tuned on four causes, on the cell's GPU"),
    "edge4_base": ("drift_eval_edge4_base.jsonl",
                   "Qwen3 1.7B, untrained, on the cell's GPU"),
}

DRIFT_SEVERE: dict[str, tuple[str, str]] = {
    "nano": ("drift_sev2.5_NVIDIA-Nemotron-3-Nano-30B-A3B.jsonl", "Nemotron 3 Nano 30B-A3B"),
    "super": ("drift_sev2.5_nemotron-3-super-120b-a12b.jsonl", "Nemotron 3 Super 120B-A12B"),
}

ARM_LABELS: dict[str, str] = {
    "nano": "Nemotron Nano 30B, every call",
    "lightning": "Nemotron 3.5 Lightning, every call",
    "super": "Nemotron Super 120B, every call",
    "ultra": "Nemotron Ultra 550B, every call",
    "routed": "Routing, what the cell does",
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
        # The two ways to be wrong cost different things: an extra look is a
        # robot move, a blind accept is a part welded on a view that a look
        # would have fixed. Before the decision set had both answers in it, only
        # the first could ever happen.
        arm["extra_looks"] = arm.get("extra_looks", 0) + (
            1 if row["said_look_again"] and not row["should_look_again"] else 0)
        arm["blind_accepts"] = arm.get("blind_accepts", 0) + (
            1 if row["should_look_again"] and not row["said_look_again"] else 0)
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
            "extra_looks": arm.get("extra_looks", 0),
            "blind_accepts": arm.get("blind_accepts", 0),
            "looks_per_decision": looks / arm["n"],
            "median_latency_ms": _median(arm["latency"]),
            "total_latency_s": sum(v for v in arm["latency"] if v) / 1000.0,
            "median_tokens": _median(arm["tokens"]),
            "gain_mm": arm["gain_mm"],
            "cost_mm": arm["cost_mm"],
        })
    order = ["routed", "nano", "lightning", "super", "ultra"]
    summary.sort(key=lambda a: order.index(a["arm"]) if a["arm"] in order else 99)

    # What the decision set is made of. A replayed point is one the campaign
    # actually reached; a first-view point is a capture seen first from one of
    # its five angles, and the ones where that view failed to measure are the
    # cases where a look is needed.
    kinds = {r.get("kind", "replayed") for r in rows}
    first_arm = next(iter(arms)) if arms else None
    own = [r for r in rows if r["arm"] == first_arm]
    points = {
        "n": len(own),
        "replayed": sum(1 for r in own if r.get("kind", "replayed") == "replayed"),
        "first_view": sum(1 for r in own if r.get("kind") == "first-view"),
        "look_needed": sum(1 for r in own if r["should_look_again"]),
        "kinds": sorted(kinds),
    }

    # A per-case grid, so the disagreements are visible rather than averaged away.
    # Trials are ints for replayed points and "part@angle" strings for first-view
    # ones, so they are ordered as text with the replayed ones first.
    def trial_key(item):
        trial = str(item[0])
        return ("@" in trial, trial.zfill(8) if "@" not in trial else trial)
    cases = sorted({(r["trial"], r["should_look_again"]) for r in rows}, key=trial_key)
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

    return {"available": True, "arms": summary, "grid": grid, "points": points}


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
                "model": registry.display(label),
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
    four_way = load(DRIFT_EDGE_FOUR)
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
        "four_way": four_way,
    }


# ------------------------------------------------------- study 5: the post-weld eye

def inspection_study() -> dict[str, Any]:
    """Named defects scored against the defect that was injected on purpose."""
    rows = cell_inspection_rows()
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
        "model": registry.display(rows[0].get("model")),
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


def _score_inspection(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    """Score one model's fifty verdicts, keeping a refusal apart from a wrong answer.

    A model may answer "uncertain", which the campaign records as calling no
    defect. Counting that as a correct pass on a clean bead flattered the models
    that refuse most; counting it as a miss on a faulty bead punished them for
    honesty. So a declined bead is scored in neither column. Accuracy is over the
    beads the model judged, and how many it judged is reported beside it.
    """
    judged = [r for r in rows if r.get("verdict") != "uncertain"]
    declined = len(rows) - len(judged)
    clean = [r for r in judged if r["injected"] == "none"]
    faulty = [r for r in judged if r["injected"] != "none"]
    called_any = {r["called"] for r in judged}
    return {
        "model": registry.display(label),
        "n": len(rows),
        "judged": len(judged),
        "declined": declined,
        "named_correct": sum(1 for r in judged if r["correct"]),
        "binary_correct": sum(1 for r in judged
                              if (r["injected"] == "none") == (r["called"] == "none")),
        "false_alarms": sum(1 for r in clean if r["called"] != "none"),
        "missed": sum(1 for r in faulty if r["called"] == "none"),
        "clean": len(clean),
        "faulty": len(faulty),
        "defects_named": len(called_any - {"none"}),
        "never_says_defect": bool(judged) and called_any == {"none"},
        "retried_with_room": sum(1 for r in rows if r.get("retried_with_room")),
        "median_latency_ms": _median([r.get("latency_ms") for r in rows]),
    }


VARIANT_LABELS = {
    ("as_written", False, 1): "prompt as first written",
    ("rendering", False, 1): "told it is a rendering",
    ("rendering", True, 1): "told it is a rendering, shown the bead up close",
    ("rendering", True, 2): "and a hole defined apart from a pore",
}


def _variant_of(row: dict[str, Any]) -> tuple[str, bool, int]:
    return (row.get("framing") or "as_written", bool(row.get("crop")),
            int(row.get("definitions") or 1))


def vision_study() -> dict[str, Any]:
    """Every model that will take an image, on the same fifty beads, under each prompt.

    The files are found rather than listed, so a model scored later is in the
    comparison the moment its run finishes; each row carries the model and the
    prompt it was run with. The first prompt called the picture a photograph
    and offered "uncertain" as a way out, and the stronger models took it: the
    picture is a rendering and they said so. So the same fifty beads were run
    again under a prompt that tells every model what it is looking at, with
    and without the bead shown up close. All three are reported; `arms` is the
    variant the cell runs with.
    """
    by_variant: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for path in sorted(DATA_DIR.glob("inspections*.jsonl")):
        rows = _read(path.name)
        if not rows or "injected" not in rows[0]:
            continue
        variant = _variant_of(rows[0])
        by_variant.setdefault(variant, []).append(
            _score_inspection(rows, rows[0].get("model", path.stem)))
    if not by_variant:
        return {"available": False}
    cell = nebius_vision_label()
    variants = []
    for key in VARIANT_LABELS:
        arms = by_variant.get(key)
        if not arms:
            continue
        arms.sort(key=lambda a: (a["model"] != cell, -a["judged"], -a["named_correct"]))
        variants.append({"framing": key[0], "crop": key[1], "definitions": key[2],
                         "label": VARIANT_LABELS[key], "arms": arms})
    wanted = cell_variant()
    running = wanted if wanted in by_variant else next(iter(by_variant))
    current = next(v for v in variants
                   if (v["framing"], v["crop"], v["definitions"]) == running)
    return {"available": True, "arms": current["arms"], "cell_model": cell,
            "cell_variant": current["label"], "variants": variants}


def cell_variant() -> tuple[str, bool, int]:
    """The prompt, framing and definitions the cell inspects with, from the same settings."""
    return (inspector.default_framing(), inspector.bead_window() is not None,
            inspector.DEFINITIONS_VERSION)


def cell_inspection_rows() -> list[dict[str, Any]]:
    """The fifty verdicts from the model and prompt the cell inspects with."""
    cell = nebius_vision_label()
    for path in sorted(DATA_DIR.glob("inspections*.jsonl")):
        rows = _read(path.name)
        if (rows and "injected" in rows[0] and _variant_of(rows[0]) == cell_variant()
                and registry.display(rows[0].get("model", "")) == cell):
            return rows
    return _read("inspections.jsonl")


def nebius_vision_label() -> str:
    """The inspector the cell runs with, as the studies name it."""
    return registry.display(nebius.vision_model())


def vision_field_study() -> dict[str, Any]:
    """Which of the account's models will take an image at all.

    One clean bead and one with a hole burnt through it, through the real
    inspection path, for every model the key can call. The platform refuses the
    image outright for most of them. Of those that accept it, some judge, some
    say they cannot tell from a rendering, and that split is what decides which
    are worth the fifty-bead run above.
    """
    rows = _read("vision_field.jsonl")
    if not rows:
        return {"available": False}
    models = []
    for r in rows:
        clean, defect = r.get("clean", {}), r.get("defect", {})
        if "error" in clean and "error" in defect:
            outcome = "refused" if clean["error"] == "no image input" else "error"
        else:
            verdicts = {clean.get("verdict"), defect.get("verdict")}
            outcome = "declined" if verdicts <= {"uncertain", None} else "judged"
        models.append({
            "model": registry.display(r["model"]),
            "raw_model": r["model"],
            "outcome": outcome,
            "clean": clean.get("called") or clean.get("error"),
            "defect": defect.get("called") or defect.get("error"),
            "seconds": r.get("seconds"),
        })
    counts = {k: sum(1 for m in models if m["outcome"] == k)
              for k in ("judged", "declined", "refused", "error")}
    order = {"judged": 0, "declined": 1, "error": 2, "refused": 3}
    models.sort(key=lambda m: (order[m["outcome"]], m["model"]))
    return {"available": True, "n": len(models), "counts": counts, "models": models}


# The sweep is run twice: once scoring the slice measurements alone, once with
# each face fitted from every point it has. The pair is the finding.
ROBUSTNESS_ARMS: dict[str, tuple[str, str]] = {
    "slices": ("robustness_slices.jsonl", "measured slice by slice"),
    "refined": ("robustness_refined.jsonl", "faces fitted from all their points"),
}



def _mean_holds(row: dict[str, Any]) -> bool:
    """The path is broadly on the joint."""
    return bool(row.get("weldable"))


def _edge(ordered: list[dict[str, Any]], passes) -> dict[str, Any]:
    """Where along a swept axis a criterion stops being met."""
    held = [r["value"] for r in ordered if passes(r)]
    failed = [r for r in ordered if not passes(r)]
    return {
        "holds_to": max(held) if held else None,
        "fails_at": failed[0]["value"] if failed else None,
        "fails_by": ("nothing measurable" if failed and failed[0]["parts_tracked"] == 0
                     else "drifting off the joint while still reporting a path"
                     if failed else None),
    }


def robustness_study() -> dict[str, Any]:
    """Where the perception stops working, and what moving it cost.

    Every other number in this repository was measured on exact rendered depth,
    which makes them ceilings rather than results. This is the only study that
    asks the question an engineer asks first, which is not how good it is but
    where the edge is.
    """
    arms = []
    for key, (name, label) in ROBUSTNESS_ARMS.items():
        rows = _read(name)
        if not rows:
            continue
        axes: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            axes.setdefault(row["axis"], []).append(row)

        limits = {}
        for axis, curve in axes.items():
            if axis == "named":
                continue
            ordered = sorted(curve, key=lambda r: r["value"])

            # The threshold is declared on the mean because that is what was
            # fixed before the sweep ran. It is not the whole story and is not
            # presented as one: the worst station on any part is what actually
            # burns through, it degrades far faster than the mean, and it is
            # carried alongside every point on the curve so a reader sees both.
            # Putting a second threshold on it now, after seeing the data, would
            # be exactly the sort of thing this study exists to avoid.
            limits[axis] = _edge(ordered, _mean_holds)
            limits[axis]["worst_mm_when_it_fails"] = next(
                (r["worst_mm"] for r in ordered if not _mean_holds(r)), None)
            limits[axis]["worst_mm_at_best"] = ordered[0].get("worst_mm") if ordered else None
        arms.append({
            "arm": key,
            "label": label,
            "axes": {a: sorted(c, key=lambda r: r["value"]) for a, c in axes.items()
                     if a != "named"},
            "named": sorted((r for r in rows if r["axis"] == "named"),
                            key=lambda r: r["mean_mm"] if r["mean_mm"] is not None else 1e9),
            "limits": limits,
        })

    if not arms:
        return {"available": False}
    return {"available": True, "weldable_mm": WELDABLE_MM, "arms": arms}


def compensation_study() -> dict[str, Any]:
    """Welding where the joint is going to be, and how wrong the model may be.

    The interesting number is not the improvement at nominal, which any
    compensation scheme can show. It is the range of model error over which
    compensating still beats not compensating, because a distortion model on a
    real machine is calibrated rather than known.
    """
    rows = _read("compensation.jsonl")
    if not rows:
        return {"available": False}

    ordered = sorted(rows, key=lambda r: r["model_error"])
    nominal = next((r for r in ordered if r["model_error"] == 0.0), ordered[0])
    helped = [r["model_error"] for r in ordered if r["helped"]]
    weldable = [r["model_error"] for r in ordered if r["weldable"]]

    return {
        "available": True,
        "curve": ordered,
        "trials": nominal.get("trials"),
        "nominal": nominal,
        "helps_from": min(helped) if helped else None,
        "helps_to": max(helped) if helped else None,
        "weldable_from": min(weldable) if weldable else None,
        "weldable_to": max(weldable) if weldable else None,
        # Over-modelling pushes the torch past the joint; under-modelling just
        # leaves some of the walk behind. Only one of those has a hard edge.
        "breaks_by": "over-predicting how far the metal moves",
    }



CONFUSABLE = ("fixture", "camera")


def confound_study() -> dict[str, Any]:
    """A drifting camera and a drifting fixture are the same measurement.

    The seam stream cannot separate them, and no model fixes that because the
    information is not in the data. What fixes it is a reference feature bolted to
    the machine base, which only the camera can move. The study is run with and
    without it so the gain is attributable to the instrument rather than the model.
    """
    rows = _read("confound.jsonl")
    if not rows:
        return {"available": False}

    def arm(reference: bool) -> dict[str, Any]:
        got = [r for r in rows if r["reference"] is reference]
        if not got:
            return {"available": False}
        by_scenario: dict[str, dict[str, int]] = {}
        for r in got:
            slot = by_scenario.setdefault(r["scenario"], {"n": 0, "correct": 0})
            slot["n"] += 1
            slot["correct"] += 1 if r["correct"] else 0
        confusable = [r for r in got if r["scenario"] in CONFUSABLE]
        return {
            "available": True,
            "reference": reference,
            "label": "with the reference" if reference else "on the seam alone",
            "n": len(got),
            "correct": sum(1 for r in got if r["correct"]),
            "by_scenario": by_scenario,
            "confusable_correct": sum(1 for r in confusable if r["correct"]),
            "confusable_n": len(confusable),
            "median_latency_ms": _median([r.get("latency_ms") for r in got]),
            "cases": got,
        }

    blind, seeing = arm(False), arm(True)
    if not blind.get("available") or not seeing.get("available"):
        return {"available": False}
    return {
        "available": True,
        "model": registry.display(rows[0].get("model")),
        "arms": [blind, seeing],
        "camera_without": blind["by_scenario"].get("camera", {}),
        "camera_with": seeing["by_scenario"].get("camera", {}),
    }


def error_budget_study() -> dict[str, Any]:
    """Everything between the joint and the torch, not just the link we measured.

    Computed rather than read from a file, because it is an argument about which
    terms exist and how they combine rather than the output of a run.
    """
    from cell import error_budget

    return {"available": True} | error_budget.summary()


HARD_FAMILY_ORDER = ("clear", "marginal", "lucky_noise", "subtle_camera", "mixed")

HARD_ARM_LABELS = {
    "rule": "the threshold as arithmetic",
    "cloud": "120B, threshold stated",
    "cloud_blind": "120B, threshold withheld",
    "edge": "1.7B trained, on the cell",
}


# The 64 case set, eight per family and scenario pair over eight seeds. The earlier
# 28 case files are not read any more and are not pooled with these: they were
# measured on a set with a hole in it, where clear + camera generated nothing at
# all, and mixing the two would average a defective test with its replacement.
# hard64_d is the edge model asked the cloud arms' exact question, with the
# fitted numbers and the boundary sentence, and once more with the sentence
# withheld. The edge arm in hard64_c was asked a different, poorer question and
# is superseded by it.
HARD_FILES = ("hard64_a.jsonl", "hard64_b.jsonl", "hard64_c.jsonl", "hard64_d.jsonl")

# Only the arms that are not a model in their own right need naming here. Every
# actual model goes through the registry, so a label used in one result file and a
# different label for the same model in another both come out as one name.
HARD_ARM_LABELS = {
    "rule": "the threshold as arithmetic",
    "cloud": "Nemotron 3 Super 120B",
    "cloud_blind": "Nemotron 3 Super 120B, boundary withheld",
    "edge": "Qwen3 1.7B trained, on the cell",
    "edge_blind": "Qwen3 1.7B trained, boundary withheld",
}


def hard_arm_label(arm: str) -> str:
    """One name per arm, whatever the result file happened to call it."""
    if arm in HARD_ARM_LABELS:
        return HARD_ARM_LABELS[arm]
    return registry.display(arm)


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for paired binary outcomes.

    b and c are the discordant counts: cases one arm got right and the other did
    not, each way round. Concordant cases carry no information about which arm is
    better and are correctly ignored.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _significance_vs_rule(passes: dict[str, list[list[dict[str, Any]]]],
                          rule_label: str) -> dict[str, dict[str, Any]]:
    """Whether each arm's margin over the fixed rule is bigger than the noise.

    Measured on ONE pass per arm rather than on the pooled passes. Pooling looked
    tempting and was wrong: the same cases repeat in every pass, so the discordant
    counts double and the p value falls by a factor that reflects nothing but how
    many times the test was run. Doing it that way turned p = 0.07 into p = 0.004
    here, which would have been a claim of significance manufactured out of a
    rerun.
    """
    rule_runs = passes.get(rule_label)
    if not rule_runs:
        return {}
    rule = {(r["family"], r["seed"], r["truth"]): bool(r["correct"])
            for r in rule_runs[0]}
    out = {}
    for label, runs in passes.items():
        if label == rule_label:
            continue
        got = {(r["family"], r["seed"], r["truth"]): bool(r["correct"]) for r in runs[0]}
        shared = sorted(set(rule) & set(got))
        if not shared:
            continue
        b = sum(1 for k in shared if got[k] and not rule[k])
        c = sum(1 for k in shared if rule[k] and not got[k])
        out[label] = {"n": len(shared), "better": b, "worse": c,
                      "p": _mcnemar_exact(b, c)}
    return out


def hard_study() -> dict[str, Any]:
    """The same question on cases built to sit ON the decision boundary.

    Every other drift score in this project is measured on runs generated by
    injecting a fault and checking it showed, so none of them is close to the
    boundary and a fixed rule cannot fail. This set is built the other way, by
    rejection sampling into chosen bands, and it is the only drift evidence here
    that separates a diagnostician from a ruler.

    Passes are pooled rather than cherry-picked, and each arm carries its per-run
    scores, because at temperature zero the same model on the same cases has come
    back 16, 16, 17 and 18. A gap of one or two between two models is inside that
    spread and means nothing. A run in which the endpoint returned errors rather
    than answers is dropped: a model that refuses an API option or was addressed by
    the wrong identifier is not a model that cannot diagnose, and one such run was
    nearly published here as a score of zero.
    """
    passes: dict[str, list[list[dict[str, Any]]]] = {}
    for name in HARD_FILES:
        rows = _read(name)
        # The edge arm in hard64_c was asked a poorer question than the cloud
        # arms beside it. hard64_d asks it the same one and replaces that arm.
        if name == "hard64_c.jsonl" and (DATA_DIR / "hard64_d.jsonl").is_file():
            rows = [r for r in rows if r.get("arm") != "edge"]
        if not rows:
            continue
        grouped: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(str(r.get("arm")), []).append(r)
        for arm, got in grouped.items():
            if any(str(r.get("said", "")).startswith("error") for r in got):
                continue
            passes.setdefault(hard_arm_label(arm), []).append(got)
    if not passes:
        return {"available": False}

    families = [f for f in HARD_FAMILY_ORDER
                if any(r.get("family") == f
                       for runs in passes.values() for got in runs for r in got)]
    arms = []
    for label, runs in passes.items():
        flat = [r for got in runs for r in got]
        by_family = {}
        for f in families:
            in_f = [r for r in flat if r["family"] == f]
            if in_f:
                by_family[f] = {"n": len(in_f),
                                "correct": sum(1 for r in in_f if r["correct"])}
        said: dict[str, int] = {}
        for r in flat:
            said[str(r.get("said"))] = said.get(str(r.get("said")), 0) + 1
        arms.append({
            "label": label,
            "runs": len(runs),
            "per_run": [sum(1 for r in got if r["correct"]) for got in runs],
            "cases_per_run": len(runs[0]),
            "n": len(flat),
            "correct": sum(1 for r in flat if r["correct"]),
            "rate": sum(1 for r in flat if r["correct"]) / len(flat),
            "by_family": by_family,
            "said": said,
            "fell_back": any(r.get("effort") == "default" for r in flat),
            "median_latency_ms": _median([r.get("ms") for r in flat]),
        })
    arms.sort(key=lambda a: -a["rate"])
    spread = max((max(a["per_run"]) - min(a["per_run"])
                  for a in arms if len(a["per_run"]) > 1), default=0)
    rule_label = HARD_ARM_LABELS["rule"]
    against_rule = _significance_vs_rule(passes, rule_label)
    for a in arms:
        a["vs_rule"] = against_rule.get(a["label"])
    seeds = sorted({r.get("seed") for runs in passes.values()
                    for got in runs for r in got if r.get("seed") is not None})
    return {"available": True, "families": list(families), "arms": arms,
            "cases": arms[0]["cases_per_run"], "worst_spread": spread,
            "seeds": len(seeds), "vs_rule": against_rule}


def view_prompt_study() -> dict[str, Any]:
    """Were the larger models over-cautious, or obeying an instruction?

    The view-planning prompt said to accept a single view only if its signals were
    "excellent", and never defined excellent. Every part starts with one view, so
    read literally the prompt asks for a second look every time. This runs the same
    decisions with that one sentence given a number, and the scores move enough to
    retire the over-caution reading.
    """
    rows = _read("view_prompt.jsonl")
    if not rows:
        return {"available": False}

    models = []
    for model in dict.fromkeys(r["model"] for r in rows):
        entry = {"model": model, "name": registry.display(model)}
        for stated in (False, True):
            got = [r for r in rows
                   if r["model"] == model and r["threshold_stated"] is stated]
            if not got:
                continue
            entry["stated" if stated else "as_written"] = {
                "n": len(got),
                "correct": sum(1 for r in got if r["correct"]),
            }
        models.append(entry)
    return {"available": True, "models": models}


def models_study() -> dict[str, Any]:
    """Every model this project calls, once, with the job it does.

    Counting the models used to give a different answer depending on which page
    you counted from, because each study named them its own way.
    """
    rows = registry.summary()
    groups: list[dict[str, Any]] = []
    for row in rows:
        if not groups or groups[-1]["group"] != row["group"]:
            groups.append({"group": row["group"], "models": []})
        groups[-1]["models"].append(row)
    return {"available": True, "n": len(rows), "groups": groups}


def all_studies() -> dict[str, Any]:
    return {
        "seam": seam_study(),
        "locating": locating_study(),
        "router": router_study(),
        "drift": drift_study(),
        "inspection": inspection_study(),
        "vision": vision_study(),
        "vision_field": vision_field_study(),
        "robustness": robustness_study(),
        "compensation": compensation_study(),
        "confound": confound_study(),
        "hard": hard_study(),
        "view_prompt": view_prompt_study(),
        "error_budget": error_budget_study(),
        "models": models_study(),
    }
