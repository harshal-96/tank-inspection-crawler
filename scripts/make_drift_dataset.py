"""Generate the training set for the drift diagnosis, out of the cell itself.

The measured result this exists to answer: reading a run of parts and naming which
machine is going out of adjustment is the one question the small model fails. Nano
scores 4/12 by answering "press" every time, which is arithmetic rather than
diagnosis. Only the 120B reads the evidence, and it is too slow and too expensive
to ask once a part.

Nothing about that is a data problem in the usual sense, because the data is not
scarce. The cell is a simulation with a forming model in it, so a run of parts with
a known fault costs nothing to make and arrives already labelled. This writes as
many of those as asked for.

Three things keep the result honest:

  the generator is shared        cell/drift.py produces both these runs and the
                                 runs the evaluation scores against, so a model
                                 cannot be trained on one description of the
                                 problem and tested on another.

  the eval seeds are excluded    the four seeds the published evaluation landed
                                 on, and a margin either side, never appear here.

  the label has to be earned     a run is only written if its measurements say
                                 what its label says. A fault that was injected
                                 but did not clear the part-to-part scatter is
                                 dropped, and so is a control that happens to
                                 look like a fault. Part-to-part scatter here is
                                 several millimetres, so over thirty parts a
                                 least squares line picks up a visible slope
                                 often enough with nothing wrong at all; the test
                                 is the slope measured against its own scatter,
                                 not the slope.

Both channels have to agree, not just the one the fault is supposed to move. A
press run whose bottom also happens to trend is genuinely ambiguous, and training
on it would teach the model to answer past its evidence, which is the exact
failure the small model already has.

The evidence sentence in each answer is written from the run's own measured
numbers rather than by a model. That makes it a template, and the model learns the
template, which is the intended trade: the point of the sentence is to make a
verdict auditable by naming what it turned on, not to sound varied.

    python scripts/make_drift_dataset.py --train 600 --val 80
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cell.drift import (  # noqa: E402
    HELD_OUT_MARGIN,
    HELD_OUT_SEEDS,
    SCENARIOS,
    SCENARIOS_WITH_CAMERA,
    THRESHOLD_NOTE,
    TRAINING_SEED_BASE,
    VISIBLE_RATIO,
    build_stream,
    discriminators,
    is_held_out,
    prompt_for,
    stats_block,
    user_prompt,
)

# How far out a machine has gone, across the training set. The evaluation is run
# at 1.0, which is marginal, so the range covers both sides of it: faults too
# small to be sure about and faults nobody would miss.
SEVERITY_RANGE = (0.6, 2.6)

# Run lengths. The evaluation uses 30; a model that has only ever seen 30 has
# learned the shape of a table rather than the shape of a trend.
PARTS_CHOICES = (20, 24, 30, 36, 40)


def reference_clause(d: dict[str, float]) -> str:
    """What the fixed feature says, when there is one to say it."""
    if "reference_ratio" not in d:
        return ""
    moved = "has moved" if d["reference_ratio"] >= VISIBLE_RATIO else "has not moved"
    return (f" The reference on the machine base reads {d['reference_slope']:+.4f} mm "
            f"per part at {d['reference_ratio']:.1f} times scatter, so the camera "
            f"{moved}.")


def evidence_for(scenario: str, d: dict[str, float], parts: int) -> str:
    """One sentence naming the numbers the verdict turned on.

    Every sentence quotes both slopes and how far each stands out of the run's own
    scatter, because that comparison is the reasoning. A sentence that named only
    the slope would be teaching the model to read half the evidence.
    """
    tilt, shift = d["tilt_slope"], d["shift_slope"]
    t_tilt, t_shift = d["tilt_ratio"], d["shift_ratio"]
    if scenario == "camera":
        return (f"The seam appears to move bodily, {shift:+.4f} mm per part at "
                f"{t_shift:.1f} times scatter with tilt flat at {t_tilt:.1f}, which on "
                f"the seam alone is indistinguishable from a bad fixture."
                + reference_clause(d)
                + " The parts are fine; the measurement is not.")
    if scenario == "press":
        return (f"Tilt trends at {tilt:+.4f} mm per part, {t_tilt:.1f} times its own "
                f"scatter, while the bottom of the seam sits at {shift:+.4f} mm per "
                f"part and only {t_shift:.1f} times scatter, which is nothing. The top "
                f"is walking and the bottom is not, so the bend angle is going out."
                + reference_clause(d))
    if scenario == "fixture":
        return (f"The bottom of the seam drifts {shift:+.4f} mm per part, {t_shift:.1f} "
                f"times its own scatter, and tilt stays inside the noise at "
                f"{tilt:+.4f} mm per part, {t_tilt:.1f} times scatter. The whole seam "
                f"moves together, so the part is being located badly."
                + reference_clause(d))
    return (f"Tilt is {tilt:+.4f} mm per part at {t_tilt:.1f} times scatter and the "
            f"bottom is {shift:+.4f} mm per part at {t_shift:.1f} times scatter. "
            f"Neither stands out of the part-to-part variation over {parts} parts, "
            f"so there is no fault to name." + reference_clause(d))


def confidence_for(scenario: str, d: dict[str, float]) -> float:
    """How sure the answer deserves to be, from how far the signal clears the noise.

    A label with a fixed confidence teaches the model to always be that sure. The
    run carries how obvious its own fault was, so the label carries it too.
    """
    if scenario == "press":
        strength = d["tilt_ratio"] / 6.0
    elif scenario == "fixture":
        strength = d["shift_ratio"] / 6.0
    elif scenario == "camera":
        strength = d["reference_ratio"] / 6.0
    else:
        strength = 1.0 - max(d["tilt_ratio"], d["shift_ratio"]) / VISIBLE_RATIO
    return round(min(0.97, max(0.55, 0.55 + 0.42 * min(1.0, max(0.0, strength)))), 2)


def make_example(scenario: str, seed: int, parts: int, severity: float,
                 reference: bool = False) -> dict | None:
    """One labelled run, or nothing if the run does not support its own label.

    A fault that was injected but did not show is dropped rather than labelled.
    Keeping it would teach the model to name a cause with no evidence for it,
    which is the exact failure the small model already has.
    """
    rows, truth = build_stream(parts, scenario, seed, severity, reference=reference)
    d = discriminators(rows)
    tilt_shows = d["tilt_ratio"] >= VISIBLE_RATIO
    shift_shows = d["shift_ratio"] >= VISIBLE_RATIO
    ref_shows = d.get("reference_ratio", 0.0) >= VISIBLE_RATIO
    if truth == "press" and not (tilt_shows and not shift_shows and not ref_shows):
        return None
    if truth == "fixture" and not (shift_shows and not tilt_shows and not ref_shows):
        return None
    if truth == "camera" and not (shift_shows and not tilt_shows and ref_shows):
        return None
    if truth == "none" and (tilt_shows or shift_shows or ref_shows):
        return None                      # scatter that looks like a fault, unlabelled
    answer = {
        "cause": truth,
        "confidence": confidence_for(truth, d),
        "evidence": evidence_for(truth, d, parts),
    }
    return {
        "messages": [
            {"role": "system", "content": prompt_for(rows) + THRESHOLD_NOTE},
            {"role": "user", "content": user_prompt(rows) + stats_block(rows)},
            {"role": "assistant", "content": json.dumps(answer)},
        ],
        # Kept out of the messages; used for the summary and for auditing the set.
        "_meta": {"scenario": truth, "seed": seed, "parts": parts,
                  "severity": round(severity, 2),
                  "tilt_ratio": round(d["tilt_ratio"], 2),
                  "shift_ratio": round(d["shift_ratio"], 2)},
    }


def generate(count: int, rng: random.Random, seed_base: int,
             used: set[int], scenarios=SCENARIOS,
             reference: bool = False) -> list[dict]:
    """Balanced across the scenarios, and never on a held-out seed."""
    out: list[dict] = []
    attempts = 0
    while len(out) < count and attempts < count * 40:
        attempts += 1
        scenario = scenarios[len(out) % len(scenarios)]
        seed = seed_base + rng.randrange(0, 900_000)
        if seed in used or is_held_out(seed):
            continue
        parts = rng.choice(PARTS_CHOICES)
        severity = rng.uniform(*SEVERITY_RANGE)
        example = make_example(scenario, seed, parts, severity, reference=reference)
        if example is None:
            continue
        used.add(seed)
        out.append(example)
    if len(out) < count:
        raise SystemExit(f"only produced {len(out)} of {count} examples; widen the seed range")
    return out


def write(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"messages": example["messages"]}) + "\n")


def summarise(name: str, examples: list[dict]) -> None:
    counts: dict[str, int] = {}
    for e in examples:
        counts[e["_meta"]["scenario"]] = counts.get(e["_meta"]["scenario"], 0) + 1
    chars = sum(len(m["content"]) for e in examples for m in e["messages"])
    print(f"  {name:<6} {len(examples):>4} examples  "
          f"{counts}  ~{chars // 4 // 1000} k tokens")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--val", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="data/finetune")
    parser.add_argument("--with-camera", action="store_true",
                        help="Four scenarios including a drifting camera, with the "
                             "reference feature that separates it from a drifting "
                             "fixture. On the seam alone those two are identical.")
    args = parser.parse_args()

    scenarios = SCENARIOS_WITH_CAMERA if args.with_camera else SCENARIOS
    rng = random.Random(args.seed)
    used: set[int] = set()
    train = generate(args.train, rng, TRAINING_SEED_BASE, used,
                     scenarios=scenarios, reference=args.with_camera)
    val = generate(args.val, rng, TRAINING_SEED_BASE + 1_000_000, used,
                   scenarios=scenarios, reference=args.with_camera)

    out = Path(args.out)
    write(out / "drift_train.jsonl", train)
    write(out / "drift_val.jsonl", val)

    print(f"held-out evaluation seeds excluded: {HELD_OUT_SEEDS} "
          f"(+/- {HELD_OUT_MARGIN})")
    summarise("train", train)
    summarise("val", val)
    print(f"\nwrote {out / 'drift_train.jsonl'} and {out / 'drift_val.jsonl'}")

    example = train[0]
    print("\nfirst example, assistant turn:")
    print("  " + example["messages"][-1]["content"][:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
