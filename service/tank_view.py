"""What the dashboard needs to show the tank, read from the recorded runs.

The dashboard was built for the welding cell and knew nothing about the tank,
which is the half the project now leads with. Somebody opening the one
interactive thing in the repository saw the wrong project.

Everything here is read from files the scorers wrote: the manifest with the
answers, the inspection verdicts, and the probe comparison. Nothing is
recomputed and nothing is stored twice, so a page cannot drift away from the
run it claims to be showing. If a file is absent the page says so rather than
inventing a shape to draw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SURVEY = ROOT / "tank" / "data" / "survey"
SCENE = ROOT / "tank" / "data" / "tank_scene.json"

# What the inspector is allowed to say, worst first, so the page can colour a
# verdict by how much it would stop the job.
SEVERITY = ("crack", "burn_through", "lack_of_fusion", "undercut",
            "porosity", "corrosion", "none")
CAMERA_WORD = {"thinning": "corrosion"}
TRACE = 0.08


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _truth(frame: dict) -> tuple[str, list[str], list[str]]:
    """The same rule the scorer uses, so the page cannot disagree with it."""
    visible = [f for f in frame.get("faults_in_view", [])
               if f.get("visible_to_a_camera")]
    seen = [CAMERA_WORD.get(f["kind"], f["kind"]) for f in visible
            if f.get("coverage", 1.0) >= TRACE]
    grazed = [CAMERA_WORD.get(f["kind"], f["kind"]) for f in visible
              if f.get("coverage", 1.0) < TRACE]
    ranked = sorted((k for k in seen if k in SEVERITY), key=SEVERITY.index)
    return (ranked[0] if ranked else "none"), seen, grazed


def summary() -> dict[str, Any]:
    """Everything the tank pages need, in one call."""
    manifest = _load(SURVEY / "manifest.json")
    if manifest is None:
        return {"available": False,
                "why": "tank/data/survey/manifest.json is not here. Run the "
                       "capture, or clone the repository with its data."}

    answers = {r["station"]: r for r in _jsonl(SURVEY / "inspection.jsonl")}
    probing = _load(SURVEY / "probing.json")
    scene = _load(SCENE) or {}

    frames, counts = [], {"right": 0, "missed": 0, "false_alarm": 0,
                          "misnamed": 0, "declined": 0}
    for frame in manifest["frames"]:
        station = frame["station"]
        truth, seen, grazed = _truth(frame)
        row = answers.get(station)
        called = row["called"] if row else None
        verdict = row["verdict"] if row else None
        listed = (row.get("faults") if row else None) or (
            [] if called in (None, "none") else [called])

        outcome = "unjudged"
        if row is not None:
            if verdict == "uncertain":
                outcome = "declined"
            elif called == truth:
                outcome = "right"
            elif truth == "none":
                outcome = "sliver" if called in grazed else "false_alarm"
            elif called == "none":
                outcome = "missed"
            else:
                outcome = "misnamed"
        if outcome in counts:
            counts[outcome] += 1

        probes = frame.get("probe_candidates", [])
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
            "probe_rejectable": sum(1 for c in probes if c.get("rejectable")),
            "thinnest_mm": min((c["true_thickness_mm"] for c in probes),
                               default=None),
        })

    points = [c for f in manifest["frames"] for c in f.get("probe_candidates", [])]
    rejectable = [c for c in points if c.get("rejectable")]

    # The contrast the whole project rests on, taken from the scene rather
    # than asserted: the patch a camera can see against the one it cannot.
    patches = [d for d in scene.get("defects", []) if d.get("kind") == "thinning"]
    visible_patch = next((d for d in patches if d.get("visible")), None)
    hidden_worst = min((d for d in patches if not d.get("visible")),
                       key=lambda d: d.get("remaining_fraction", 1.0), default=None)

    planners = []
    for p in (probing or {}).get("planners", []):
        planners.append({
            "label": p["label"],
            "probes": p.get("spent"),
            "found": len(p.get("rejectable_found", [])),
            "thinned_found": p.get("thinned_found"),
            "attempt": p.get("attempt"),
        })

    return {
        "available": True,
        "scored": bool(answers),
        "stations": len(manifest["frames"]),
        "resolution": manifest.get("resolution"),
        "run": manifest.get("survey_run", {}),
        "tank": manifest.get("tank", {}),
        "acceptance": manifest.get("acceptance", {}),
        "evaluation_scene": manifest.get("evaluation_scene"),
        "frames": frames,
        "counts": counts,
        "probe": {
            "points": len(points),
            "budget_points": (probing or {}).get("budget", 2) * len(manifest["frames"]),
            "thinned": sum(1 for c in points if c.get("thinned_by")),
            "rejectable": len(rejectable),
            "rejectable_under_a_clean_frame": sum(
                1 for f in manifest["frames"]
                for c in f.get("probe_candidates", [])
                if c.get("rejectable") and _truth(f)[0] == "none"),
            "planners": planners,
        },
        "contrast": {
            "visible": None if visible_patch is None else {
                "id": visible_patch["id"],
                "loss_mm": visible_patch.get("loss_mm"),
                "remaining_mm": visible_patch.get("remaining_mm"),
                "remaining_fraction": visible_patch.get("remaining_fraction"),
                "rejectable": visible_patch.get("rejectable"),
            },
            "hidden": None if hidden_worst is None else {
                "id": hidden_worst["id"],
                "loss_mm": hidden_worst.get("loss_mm"),
                "remaining_mm": hidden_worst.get("remaining_mm"),
                "remaining_fraction": hidden_worst.get("remaining_fraction"),
                "rejectable": hidden_worst.get("rejectable"),
            },
            "limit_fraction": manifest.get("acceptance", {})
                                      .get("thinning", {}).get("limit_fraction"),
        },
    }
