"""How much of the sweep's score belongs to where the hidden patches were put.

    python tank/layout_sweep.py                          # 1000 layouts, seed 1
    python tank/layout_sweep.py --layouts 200 --seed 3   # a different draw
    python tank/layout_sweep.py --out SCRATCH/layouts.json

The five patches of hidden wall loss on the survey run were placed by hand,
after the first result on the four point scene, and every one of them sits well
off the seam. A rule that covers the band in rotation is the kind of planner
such a layout rewards. This script asks how much of the sweep's 16 of 28
survives when the same patches are put somewhere else.

One layout, as drawn here:

- the same number of hidden patches as the shipped run (the scene's thinning
  patches that a camera cannot see and that thin a probe point on the run);
- each centre uniform along the survey run's arc (survey_run in the scene) and
  uniform up and down the wall within the probe's reach of the seam (the
  largest probe offset either way);
- each radius uniform between the smallest and largest radius of the shipped
  hidden patches, and each loss uniform between their smallest and largest
  loss, all read from tank/data/tank_scene.json and printed at the start;
- drawn independently, so two patches may overlap, and where they do their
  losses add, which is what tank.thickness does with the shipped ones;
- everything else in the scene, the visible patch included, left where it is.

The 168 probe answers are recomputed in memory with tank.thickness.thickness_mm
at the manifest's probe positions, and only the arithmetic arms are scored: the
sweep with its shipped tie-break, on the seam, the mean of random picks, and
the oracle. No model is asked anything. Before drawing, the shipped layout is
put through the same code and must reproduce the manifest's answers and the
finds recorded in probing.json, so the numbers below are about the planner
that was measured and not about a copy of it.

Nothing is written under tank/data. The recorded results stay what they were.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tank.thickness import thickness_mm  # noqa: E402

DATA = ROOT / "tank" / "data"
SURVEY = DATA / "survey"
SCENE = DATA / "tank_scene.json"


def planner_constants(*names) -> dict:
    """Constants read out of probe_planner.py without importing it.

    probe_planner imports the model client, and this script must not be able
    to reach a model. Copying the numbers would let them drift, so the source
    is parsed for the literal assignments, as tank.adhesion does for the scene.
    """
    source = (ROOT / "tank" / "probe_planner.py").read_text(encoding="utf-8")
    out = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    out[target.id] = ast.literal_eval(node.value)
    missing = set(names) - set(out)
    if missing:
        raise KeyError(f"not found in probe_planner.py: {sorted(missing)}")
    return out


_P = planner_constants("WATCH_FRACTION", "DEFAULT_BUDGET")
WATCH_FRACTION = _P["WATCH_FRACTION"]
DEFAULT_BUDGET = _P["DEFAULT_BUDGET"]


# ----------------------------------------------------------- the arithmetic arms


def sweep(stations, budget: int) -> list[tuple[int, float]]:
    """probe_planner.sweep_rule, over a run whose answers are already known.

    Chase an offset that read at or below the watch level in the last two
    stations, then take the offsets longest without a reading, ties broken
    nearest the seam and below it first. Returns the (station, offset) pairs
    measured.
    """
    readings: list[tuple[int, float, float]] = []      # station, offset, fraction
    measured = []
    for station, points in stations:
        offsets = [p["offset_m"] for p in points]
        picked: list[float] = []
        for s, o, fraction in reversed(readings):
            if station - s > 2:
                break
            if fraction <= WATCH_FRACTION and o not in picked and len(picked) < budget:
                picked.append(o)
        last = {}
        for s, o, _ in readings:
            last[o] = s

        def staleness(offset, station=station, last=last):
            return (-(10 ** 6) if offset not in last else -(station - last[offset]),
                    abs(offset))

        for offset in sorted(offsets, key=staleness):
            if len(picked) >= budget:
                break
            if offset not in picked:
                picked.append(offset)
        for offset in picked[:budget]:
            p = next(q for q in points if abs(q["offset_m"] - offset) < 1e-6)
            readings.append((station, offset, p["fraction"]))
            measured.append((station, offset))
    return measured


def on_the_seam(stations, budget: int) -> list[tuple[int, float]]:
    """Offset 0.0, then the nearest offsets, below the seam first."""
    out = []
    for station, points in stations:
        offsets = [p["offset_m"] for p in points]
        picked = [o for o in offsets if abs(o) < 1e-6]
        picked += [o for o in sorted(offsets, key=abs) if o not in picked]
        out.extend((station, o) for o in picked[:budget])
    return out


def score(stations, measured, budget: int) -> dict:
    by_place = {(s, p["offset_m"]): p for s, points in stations for p in points}
    hits = [by_place[m] for m in measured if by_place[m]["rejectable"]]
    return {"found": len(hits),
            "patches": sorted({i for p in hits for i in p["thinned_by"]})}


def references(stations, budget: int) -> dict:
    """Random picks on average, and the most a planner could find."""
    mean = best = 0.0
    for _, points in stations:
        k = sum(1 for p in points if p["rejectable"])
        take = min(budget, len(points))
        mean += take * k / len(points)
        best += min(take, k)
    return {"chance_mean": mean, "oracle": int(best)}


# --------------------------------------------------------------- the layouts


def answers(manifest, defects, limit: float):
    """The probe points at every station, recomputed for these defects."""
    stations = []
    for frame in manifest["frames"]:
        points = []
        for c in frame["probe_candidates"]:
            left, took = thickness_mm(defects, c["theta_rad"], c["z_m"])
            fraction = left / c["nominal_mm"]
            points.append({"offset_m": c["offset_m"], "thickness_mm": left,
                           "fraction": round(fraction, 4), "thinned_by": took,
                           "rejectable": round(fraction, 4) < limit})
        stations.append((frame["station"], points))
    return stations


def hidden_on_run(scene, manifest) -> list[dict]:
    """The scene's hidden patches that thin a probe point on the run."""
    touched = {i for f in manifest["frames"] for c in f["probe_candidates"]
               for i in c["thinned_by"]}
    return [d for d in scene["defects"]
            if d["kind"] == "thinning" and not d["visible"] and d["id"] in touched]


def draw(rng: random.Random, template: list[dict], run: dict, reach: float,
         radius: tuple[float, float], loss: tuple[float, float]) -> list[dict]:
    out = []
    for d in template:
        out.append({"id": d["id"], "kind": "thinning",
                    "theta_rad": rng.uniform(run["theta_from_rad"], run["theta_to_rad"]),
                    "z_m": run["z_m"] + rng.uniform(-reach, reach),
                    "patch_radius_m": rng.uniform(*radius),
                    "loss_mm": rng.uniform(*loss)})
    return out


def percentile_of(value: float, values: list[float]) -> tuple[float, float]:
    """Share of values below, and at or below, this one."""
    n = len(values)
    return (sum(1 for v in values if v < value) / n,
            sum(1 for v in values if v <= value) / n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layouts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--out", default=None,
                        help="write the per-layout results here. Refused anywhere "
                             "under tank/data")
    args = parser.parse_args()

    if args.layouts < 200:
        print("at least 200 layouts, or the tails of the distribution are noise")
        return 2
    out_path = Path(args.out).resolve() if args.out else None
    if out_path is not None and out_path.is_relative_to(DATA.resolve()):
        print(f"refusing to write {out_path}: nothing from this script goes under "
              "tank/data, which holds the recorded results")
        return 2

    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    manifest = json.loads((SURVEY / "manifest.json").read_text(encoding="utf-8"))
    limit = manifest["acceptance"]["thinning"]["limit_fraction"]
    run = scene["survey_run"]
    reach = max(abs(o) for o in manifest["probe_offsets_m"])
    template = hidden_on_run(scene, manifest)
    fixed = [d for d in scene["defects"] if d not in template]
    radius = (min(d["patch_radius_m"] for d in template),
              max(d["patch_radius_m"] for d in template))
    loss = (min(d["loss_mm"] for d in template), max(d["loss_mm"] for d in template))

    # The shipped layout first, through the same code, against the record.
    shipped = answers(manifest, fixed + template, limit)
    for (station, points), frame in zip(shipped, manifest["frames"], strict=True):
        for p, c in zip(points, frame["probe_candidates"], strict=True):
            if (abs(p["thickness_mm"] - c["true_thickness_mm"]) > 1e-6
                    or p["rejectable"] != c["rejectable"]):
                raise SystemExit(f"station {station} {c['offset_m']:+.2f}: recomputed "
                                 f"{p['thickness_mm']} mm, manifest says "
                                 f"{c['true_thickness_mm']} mm. The layouts would be "
                                 "scored by different physics than the run.")
    recorded = json.loads((SURVEY / "probing.json").read_text(encoding="utf-8"))
    for label, arm in (("sweep", sweep), ("on the seam", on_the_seam)):
        row = next(p for p in recorded["planners"] if p["label"].startswith(label))
        want = sorted((s, round(o, 2)) for s, o, _ in row["rejectable_found"])
        by_place = {(s, p["offset_m"]): p for s, pts in shipped for p in pts}
        got = sorted((s, round(o, 2)) for s, o in arm(shipped, args.budget)
                     if by_place[(s, o)]["rejectable"])
        if got != want:
            raise SystemExit(f"{label}: this copy finds {got} on the shipped layout, "
                             f"probing.json recorded {want}")

    def summarise(stations) -> dict:
        rejectable = sum(1 for _, pts in stations for p in pts if p["rejectable"])
        ref = references(stations, args.budget)
        s = score(stations, sweep(stations, args.budget), args.budget)
        seam = score(stations, on_the_seam(stations, args.budget), args.budget)
        patches = sorted({i for _, pts in stations for p in pts if p["rejectable"]
                          for i in p["thinned_by"]})
        return {"rejectable": rejectable, "patches": len(patches),
                "sweep": s["found"], "sweep_patches": len(s["patches"]),
                "on_the_seam": seam["found"], "seam_patches": len(seam["patches"]),
                "chance_mean": ref["chance_mean"], "oracle": ref["oracle"]}

    as_shipped = summarise(shipped)
    rng = random.Random(args.seed)
    layouts, empty = [], 0
    for _ in range(args.layouts):
        got = summarise(answers(manifest, fixed + draw(rng, template, run, reach,
                                                       radius, loss), limit))
        if got["rejectable"] == 0:
            empty += 1
            continue
        layouts.append(got)

    print(f"{args.layouts} layouts, seed {args.seed}, {len(template)} hidden patches each "
          f"({', '.join(d['id'] for d in template)} on the shipped run)")
    print(f"  centre along the run: uniform {run['theta_from_rad']:+.3f} to "
          f"{run['theta_to_rad']:+.3f} rad; across the seam: uniform "
          f"{-reach:+.2f} to {reach:+.2f} m")
    size = (f"radius {radius[0]:.2f} m, as every shipped hidden patch"
            if radius[0] == radius[1] else
            f"radius uniform {radius[0]:.2f} to {radius[1]:.2f} m")
    print(f"  {size}; loss uniform {loss[0]:.1f} to {loss[1]:.1f} mm; "
          f"limit {limit * 100:.0f}% of nominal")
    print("  shipped layout reproduces the manifest and probing.json: yes")
    if empty:
        print(f"  {empty} layouts had no rejectable point and are left out")
    print()

    def share(row, key):
        return row[key] / row["rejectable"]

    print(f"{'share of rejectable found':>28} {'shipped':>9} {'median':>8} "
          f"{'5th':>7} {'95th':>7} {'mean':>7}  shipped percentile")
    print("-" * 92)
    for label, key in (("sweep", "sweep"), ("on the seam", "on_the_seam"),
                       ("chance mean", "chance_mean"), ("oracle", "oracle")):
        values = [share(r, key) for r in layouts]
        cut = statistics.quantiles(values, n=20, method="inclusive")
        below, at = percentile_of(share(as_shipped, key), values)
        # Random picks find budget/places of whatever is there, in any layout,
        # so a percentile for that row would only rank rounding error.
        where = ("the same in every layout" if max(values) - min(values) < 1e-9 else
                 f"{below * 100:.1f}% of layouts below, {at * 100:.1f}% at or below")
        print(f"{label:>28} {share(as_shipped, key) * 100:>8.1f}% "
              f"{statistics.median(values) * 100:>7.1f}% {cut[0] * 100:>6.1f}% "
              f"{cut[-1] * 100:>6.1f}% {statistics.fmean(values) * 100:>6.1f}%  {where}")
    print("-" * 92)
    counts = [r["sweep"] for r in layouts]
    print(f"  shipped: sweep {as_shipped['sweep']} of {as_shipped['rejectable']}, "
          f"on the seam {as_shipped['on_the_seam']}, chance mean "
          f"{as_shipped['chance_mean']:.1f}, oracle {as_shipped['oracle']}")
    print(f"  rejectable points per layout: median "
          f"{statistics.median(r['rejectable'] for r in layouts):g}, range "
          f"{min(r['rejectable'] for r in layouts)} to "
          f"{max(r['rejectable'] for r in layouts)}; sweep found median "
          f"{statistics.median(counts):g}, range {min(counts)} to {max(counts)}")
    ahead = sum(1 for r in layouts if r["sweep"] > r["chance_mean"])
    seam_ahead = sum(1 for r in layouts if r["sweep"] > r["on_the_seam"])
    every = sum(1 for r in layouts if r["sweep_patches"] == r["patches"])
    print(f"  sweep above the chance mean in {ahead} of {len(layouts)} layouts, "
          f"above on the seam in {seam_ahead}; it reached every rejectable patch "
          f"in {every}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump({"layouts": args.layouts, "seed": args.seed,
                       "budget": args.budget, "radius_m": radius, "loss_mm": loss,
                       "reach_m": reach, "theta_rad": [run["theta_from_rad"],
                                                       run["theta_to_rad"]],
                       "left_out_empty": empty, "shipped": as_shipped,
                       "results": layouts}, handle, indent=1)
            handle.write("\n")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
