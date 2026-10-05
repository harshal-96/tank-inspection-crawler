"""What the dashboard needs to show the tank, read from the recorded runs.

The dashboard was built for the welding cell and knew nothing about the tank,
which is the half the project now leads with. Somebody opening the one
interactive thing in the repository saw the wrong project.

The verdicts and the probe picks are read from the files the scorers wrote
and never asked for again: nothing here calls a model. What is worked out
here is the marking. The answer key comes from tank/truth.py, the one rule the
scorer and the tests also use, so a frame is marked the same way on this page
as in the replay. The random-pick band and the best-possible figure are read
from the scorer's baselines file; the band is recomputed here, exactly, only
when that file predates them. If a file is absent or ambiguous the page says
so rather than inventing a shape to draw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tank.truth import CAMERA_WORD, TRACE, load_answers, truth_for

ROOT = Path(__file__).resolve().parent.parent
SURVEY = ROOT / "tank" / "data" / "survey"
SCENE = ROOT / "tank" / "data" / "tank_scene.json"

# The probe comparisons that are shown. Retired runs carry a different count
# of failing points from the manifest and are skipped by that check, so a file
# moved or left behind cannot put an old scene's numbers on this page.
PROBING = ("probing.json", "probing_power.json", "probing_blind.json",
           "probing_baselines.json")
# The arithmetic arms are deterministic, so they are read from one file: the
# baselines file when it is on the current scene, else the main run.
BASELINE_FILES = ("probing_baselines.json", "probing.json")

OUTCOMES = ("right", "misnamed", "missed", "false_alarm", "sliver",
            "declined", "unjudged")


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _outcome(row: dict | None, truth: str, seen: list[str], grazed: list[str]) -> str:
    if row is None:
        return "unjudged"
    called = row["called"]
    if row["verdict"] == "uncertain":
        return "declined"
    if called == truth:
        return "right"
    if truth == "none":
        return "sliver" if called in grazed else "false_alarm"
    if called == "none":
        return "missed"
    return "misnamed"


def _per_fault(frames: list[dict], answers: dict[int, dict]) -> list[dict]:
    """Each seam or surface fault on the run, and whether any frame holding it
    named it. The same rule as per_fault in tank/score_survey.py, which this
    module does not import because that one loads the model client."""
    faults: dict[str, dict] = {}
    for frame in frames:
        row = answers.get(frame["station"])
        if row is None or row["verdict"] == "uncertain":
            continue
        named = set(row.get("faults") or
                    ([] if row["called"] == "none" else [row["called"]]))
        for f in frame["faults_in_view"]:
            if not f["visible_to_a_camera"] or f.get("coverage", 1.0) < TRACE:
                continue
            word = CAMERA_WORD.get(f["kind"], f["kind"])
            entry = faults.setdefault(f["id"], {"id": f["id"], "kind": word,
                                                "frames": [], "named_in": []})
            entry["frames"].append(frame["station"])
            if word in named:
                entry["named_in"].append(frame["station"])
    return list(faults.values())


def _chance(frames: list[dict], budget: int) -> dict[str, Any] | None:
    """What picking the probe places at random would find, exactly.

    At each station the arm picks `budget` of the reachable places with no
    information at all, so the count of failing points it lands on there is
    hypergeometric. The stations are independent, so the run total is the
    convolution of the stations. Exact, not sampled.
    """
    from math import comb

    pmf = [1.0]
    for frame in frames:
        cands = frame.get("probe_candidates", [])
        n, bad = len(cands), sum(1 for c in cands if c.get("rejectable"))
        k = min(budget, n)
        if n == 0 or k == 0:
            continue
        here = [comb(bad, j) * comb(n - bad, k - j) / comb(n, k)
                for j in range(min(bad, k) + 1)]
        out = [0.0] * (len(pmf) + len(here) - 1)
        for i, a in enumerate(pmf):
            for j, b in enumerate(here):
                out[i + j] += a * b
        pmf = out
    if len(pmf) == 1:
        return None

    def quantile(q: float) -> int:
        total = 0.0
        for value, p in enumerate(pmf):
            total += p
            if total >= q - 1e-12:
                return value
        return len(pmf) - 1

    return {"mean": round(sum(v * p for v, p in enumerate(pmf)), 3),
            "low": quantile(0.025), "high": quantile(0.975),
            "band": "central 95 percent, exact"}


def _baseline_file(rejectable_total: int) -> tuple[str | None, dict]:
    """The file the arithmetic arms are read from, and what it holds."""
    for name in BASELINE_FILES:
        data = _load(SURVEY / name)
        if data is not None and data.get("rejectable") == rejectable_total:
            return name, data
    return None, {}


def _planners(manifest: dict, rejectable_total: int) -> tuple[list[dict], list[str], int]:
    """Every probe planner on the current scene, with every attempt it had.

    A model at temperature zero still came back with different picks on
    different runs, so showing only its best attempt would flatter it. Each
    label carries all of its attempts and the page shows the range.
    """
    by_point = {}
    for f in manifest["frames"]:
        for c in f.get("probe_candidates", []):
            by_point[(f["station"], round(c["offset_m"], 4))] = c

    def patches(found) -> list[str]:
        hit = set()
        for station, offset, *_ in found:
            c = by_point.get((station, round(offset, 4)))
            if c and c.get("rejectable"):
                hit.update(c.get("thinned_by") or [])
        return sorted(hit)

    def current(name: str):
        data = _load(SURVEY / name)
        if data is None or data.get("rejectable") != rejectable_total:
            return None
        return data

    sources, budget = [], 2
    labels: dict[str, dict] = {}

    def points(found) -> set:
        return {(station, round(offset, 4)) for station, offset, *_ in found}

    def paired_p(pl: dict, rule_hits: set | None) -> float | None:
        """Exact McNemar of a model run against the sweep in the same file.

        The same pairing and the same function as tank/significance.py, which
        calls no model, so the page can only say "no run differs" when every
        run on record has been tested and none does.
        """
        if "seconds" not in pl or rule_hits is None:
            return None
        try:
            from tank.significance import mcnemar_exact
        except ImportError:
            return None
        hits = points(pl.get("rejectable_found", []))
        return round(mcnemar_exact(len(hits - rule_hits), len(rule_hits - hits)), 4)

    def add(pl: dict, source: str, rule_hits: set | None):
        found = pl.get("rejectable_found", [])
        entry = labels.setdefault(pl["label"], {
            "label": pl["label"], "uses_a_model": "seconds" in pl,
            "attempts": []})
        entry["attempts"].append({
            "found": len(found), "patches": patches(found),
            "probes": pl.get("spent"), "thinned_found": pl.get("thinned_found"),
            "paired_p": paired_p(pl, rule_hits),
            "source": source})

    baseline_from = _baseline_file(rejectable_total)[0]
    for name in PROBING:
        data = current(name)
        if data is None:
            continue
        sources.append(name)
        budget = data.get("budget", budget)
        sweep = next((pl for pl in data.get("planners", [])
                      if pl["label"].startswith("sweep (")), None)
        rule_hits = points(sweep["rejectable_found"]) if sweep else None
        for pl in data.get("planners", []):
            # A model arm took time to answer; every run of one is an attempt.
            if "seconds" in pl or name == baseline_from:
                add(pl, name, rule_hits)

    out = []
    for entry in labels.values():
        found = [a["found"] for a in entry["attempts"]]
        reached = [len(a["patches"]) for a in entry["attempts"]]
        tested = [a["paired_p"] for a in entry["attempts"]]
        entry.update(low=min(found), high=max(found), runs=len(found),
                     patches_low=min(reached), patches_high=max(reached),
                     probes=entry["attempts"][0]["probes"],
                     # Only when every attempt was tested; one untested run
                     # leaves the range unknown rather than understated.
                     paired_p=None if None in tested else [min(tested), max(tested)])
        out.append(entry)
    return out, sources, budget


# The tie-break study is two passes over 5040 orders, about two seconds, so it
# is worked out once and kept for as long as the files it reads are unchanged.
_TIEBREAKS: dict[tuple, dict | None] = {}


def _tiebreaks(manifest: dict, answers: dict[int, dict], budget: int,
               rejectable_total: int, read: tuple[Path, ...]) -> dict[str, Any] | None:
    """The coverage rule under every order of its tie-break.

    The rule's count on this page is one order out of 5040, and the order it
    ships with happens to be a good one. Without the spread the page would
    compare one lucky draw with the model's runs. This is the scorer's own
    study (tank/score_probes.py --tiebreaks), which calls no model, run on the
    same manifest and verdicts, for the rule with its full history and for the
    rule shown only what the model's prompt carries.
    """
    key = (budget, *(p.stat().st_mtime_ns if p.exists() else 0 for p in read))
    if key not in _TIEBREAKS:
        try:
            from tank.score_probes import tiebreak_study
            study = tiebreak_study(manifest["frames"], answers, budget, rejectable_total)
        except (ImportError, KeyError, SystemExit):
            # SystemExit is the study refusing because its reordered rule no
            # longer reproduces the shipped one; the page then says nothing
            # rather than describing a different rule.
            study = None
        _TIEBREAKS.clear()
        _TIEBREAKS[key] = None if study is None else {
            "orders": study["orders"],
            "shipped_found": study["shipped_found"],
            "median": study["median"],
            "mean": round(study["mean"], 2),
            "min": study["min"],
            "max": study["max"],
            "share_below_shipped": round(study["share_below_shipped"], 3),
            "models_view": {k: (round(v, 2) if k == "mean" else v)
                            for k, v in study["models_view"].items() if k != "counts"},
        }
    return _TIEBREAKS[key]


def summary() -> dict[str, Any]:
    """Everything the tank pages need, in one call."""
    manifest = _load(SURVEY / "manifest.json")
    if manifest is None:
        return {"available": False,
                "why": "tank/data/survey/manifest.json is not here. Run the "
                       "capture, or clone the repository with its data."}

    path = SURVEY / "inspection.jsonl"
    try:
        answers = load_answers(path) if path.exists() else {}
    except ValueError as err:
        # Said with the repository-relative name, not this machine's path.
        shown = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name
        return {"available": False,
                "why": f"The recorded verdicts cannot be read as one answer a "
                       f"station: {str(err).replace(str(path), shown)}"}
    scene = _load(SCENE) or {}

    frames = []
    counts = dict.fromkeys(OUTCOMES, 0)
    gate = {"right": 0, "judged": 0, "strict_right": 0,
            "named_only_a_sliver": 0, "named_a_lesser_fault_there": 0}
    under_pass = under_clean = 0
    for frame in manifest["frames"]:
        station = frame["station"]
        truth, seen, grazed = truth_for(frame)
        row = answers.get(station)
        called = row["called"] if row else None
        listed = (row.get("faults") if row else None) or (
            [] if called in (None, "none") else [called])
        outcome = _outcome(row, truth, seen, grazed)
        counts[outcome] += 1

        if outcome not in ("declined", "unjudged"):
            gate["judged"] += 1
            # The gate: did it say defect exactly when a fault was in front of
            # the lens. The strict form also asks that what it named is one of
            # those faults, so a defect call made for a sliver at the edge of
            # the frame is not credited for the fault it never mentioned.
            if (truth == "none") == (called == "none"):
                gate["right"] += 1
            if (truth == "none" and called == "none") or (
                    truth != "none" and called in seen):
                gate["strict_right"] += 1
            if outcome == "misnamed":
                if called in seen:
                    gate["named_a_lesser_fault_there"] += 1
                elif called in grazed:
                    gate["named_only_a_sliver"] += 1

        probes = frame.get("probe_candidates", [])
        failing = sum(1 for c in probes if c.get("rejectable"))
        if called == "none":
            under_pass += failing
        if truth == "none":
            under_clean += failing
        frames.append({
            "station": station,
            "theta_rad": frame.get("theta_rad"),
            "rgb": frame.get("rgb"),
            "truth": truth,
            "in_view": seen,
            "grazed": grazed,
            "called": called,
            "faults": listed,
            "outcome": outcome,
            "reasoning": (row or {}).get("reasoning", ""),
            "latency_ms": (row or {}).get("latency_ms"),
            # The probe row under this station: how many places the arm could
            # reach, how many are thinned, and how many fail outright.
            "probe_points": len(probes),
            "probe_thinned": sum(1 for c in probes if c.get("thinned_by")),
            "probe_rejectable": failing,
            "thinnest_mm": min((c["true_thickness_mm"] for c in probes),
                               default=None),
        })
    gate["declined"] = counts["declined"]
    gate["slivers"] = counts["sliver"]

    points = [c for f in manifest["frames"] for c in f.get("probe_candidates", [])]
    rejectable = [c for c in points if c.get("rejectable")]

    # The contrast the whole project rests on, taken from the scene rather
    # than asserted: the patches a camera can see against the ones it cannot.
    # Only patches on this run count, meaning ones some probe point sits in or
    # some frame points at; the scene holds others on seams nobody surveyed.
    on_run = {i for c in points for i in (c.get("thinned_by") or [])}
    on_run |= {f["id"] for fr in manifest["frames"] for f in fr["faults_in_view"]}
    defects = manifest.get("defects") or scene.get("defects", [])
    patches = [d for d in defects if d.get("kind") == "thinning" and d["id"] in on_run]
    visible = [d for d in patches if d.get("visible")]
    hidden = [d for d in patches if not d.get("visible")]
    visible_ids = {d["id"] for d in visible}

    def patch(d):
        return None if d is None else {
            "id": d["id"],
            "loss_mm": d.get("loss_mm"),
            "remaining_mm": d.get("remaining_mm"),
            "remaining_fraction": d.get("remaining_fraction"),
            "rejectable": bool(d.get("rejectable")),
        }

    def thinnest(group):
        return min(group, key=lambda d: d.get("remaining_fraction", 1.0), default=None)

    planners, sources, budget = _planners(manifest, len(rejectable))
    faults = _per_fault(manifest["frames"], answers)
    # The scorer writes the random-pick band and the best-possible bound beside
    # its arms. Use its numbers when it has; the band is recomputed only for a
    # baseline file written before the scorer recorded one.
    baseline = _baseline_file(len(rejectable))[1]
    recorded = baseline.get("chance") or {}
    if "mean" in recorded and len(recorded.get("band_95") or []) == 2:
        chance = {"mean": recorded["mean"], "low": recorded["band_95"][0],
                  "high": recorded["band_95"][1],
                  "band": "central 95 percent, exact, as the scorer recorded it"}
    else:
        chance = _chance(manifest["frames"], budget)
    oracle = baseline.get("oracle") or {}

    return {
        "available": True,
        "scored": bool(answers),
        # Which model judged the pictures, read off the verdict rows, so the
        # page names the inspector it is scoring rather than assuming one.
        "inspector": sorted({r["model"].split("/")[-1] for r in answers.values()
                             if r.get("model")}),
        "stations": len(manifest["frames"]),
        "resolution": manifest.get("resolution"),
        "run": manifest.get("survey_run", {}),
        "tank": manifest.get("tank", {}),
        "acceptance": manifest.get("acceptance", {}),
        "evaluation_scene": manifest.get("evaluation_scene"),
        "frames": frames,
        "counts": counts,
        "gate": gate,
        "faults_on_the_run": {
            "total": len(faults),
            "named": sum(1 for f in faults if f["named_in"]),
            "faults": faults,
        },
        "probe": {
            "points": len(points),
            "budget": budget,
            "budget_points": budget * len(manifest["frames"]),
            "thinned": sum(1 for c in points if c.get("thinned_by")),
            "rejectable": len(rejectable),
            # Two different statements, so two numbers. Under a frame the
            # inspector passed is what the camera's verdict would have left
            # unprobed; under a frame with no visible fault is what no camera
            # could have flagged, whatever it said.
            "rejectable_under_a_pass": under_pass,
            "rejectable_under_a_clean_frame": under_clean,
            "rejectable_on_a_visible_patch": sum(
                1 for c in rejectable
                if set(c.get("thinned_by") or []) & visible_ids),
            "patches_with_failing_points": len(
                {i for c in rejectable for i in (c.get("thinned_by") or [])}),
            "chance": chance,
            "best_possible": oracle.get("found"),
            "tiebreaks": _tiebreaks(manifest, answers, budget, len(rejectable),
                                    (SURVEY / "manifest.json", path)),
            "planners": planners,
            "sources": sources,
        },
        "contrast": {
            "visible": patch(thinnest(visible)),
            "hidden": patch(thinnest(hidden)),
            "visible_count": len(visible),
            "visible_failing": sum(1 for d in visible if d.get("rejectable")),
            "hidden_count": len(hidden),
            "hidden_failing": sum(1 for d in hidden if d.get("rejectable")),
            "limit_fraction": manifest.get("acceptance", {})
                                      .get("thinning", {}).get("limit_fraction"),
        },
    }
