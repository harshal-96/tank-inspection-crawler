"""The training set for the drift diagnosis, and the things that keep it honest.

A fine-tuned model is only worth the evaluation behind it, and the two ways this
one could be quietly worthless are both mechanical: training on the seeds it is
scored against, and training on runs whose measurements do not support their own
label. Both are checked here rather than being asserted in a README.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cell.drift import (
    HELD_OUT_SEEDS,
    SCENARIOS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_REFERENCE,
    VISIBLE_RATIO,
    build_stream,
    discriminators,
    is_held_out,
    prompt_for,
    slope_ratio,
    user_prompt,
)

ROOT = Path(__file__).resolve().parent.parent


def test_a_run_is_reproducible_from_its_seed():
    """The evaluation and the training set have to see the same numbers."""
    a, _ = build_stream(30, "press", 4242, 1.0)
    b, _ = build_stream(30, "press", 4242, 1.0)
    assert a == b


def test_press_tilts_and_fixture_shifts():
    """The discriminator has to actually discriminate, or nothing else matters."""
    press = discriminators(build_stream(30, "press", 20001, 1.0)[0])
    fixture = discriminators(build_stream(30, "fixture", 20001, 1.0)[0])

    assert press["tilt_ratio"] > VISIBLE_RATIO
    assert press["shift_ratio"] < VISIBLE_RATIO
    assert fixture["shift_ratio"] > VISIBLE_RATIO
    assert fixture["tilt_ratio"] < VISIBLE_RATIO


def test_a_slope_is_measured_against_its_own_scatter():
    """A straight line has an enormous ratio; noise around zero has a small one."""
    ramp = [float(i) for i in range(30)]
    noise = [0.0, 1.0, -1.0, 0.5, -0.5] * 6

    ramp_slope, ramp_ratio = slope_ratio(ramp)
    _, noise_ratio = slope_ratio(noise)

    assert ramp_slope == pytest.approx(1.0)
    assert ramp_ratio > 100
    assert noise_ratio < 2.0


def test_held_out_seeds_are_fenced_with_a_margin():
    for seed in HELD_OUT_SEEDS:
        assert is_held_out(seed)
        assert is_held_out(seed + 5)
        assert is_held_out(seed - 5)
    assert not is_held_out(20_000)


def test_the_prompt_is_shared_rather_than_retyped():
    """A trained model given a better prompt has not been compared to its baseline.

    The guarantee used to be that the evaluation imported the prompt by name. It is
    now stronger: the prompt is chosen from the rows themselves, so a model cannot
    be told about a reference column its table does not carry, or shown one nobody
    described. Checked on the function every evaluation calls.
    """
    import scripts.eval_drift_edge as edge  # noqa: PLC0415

    assert edge.prompt_for is prompt_for

    plain, _ = build_stream(8, "press", 4242)
    assert prompt_for(plain) is SYSTEM_PROMPT
    assert "reference" not in prompt_for(plain).lower()

    instrumented, _ = build_stream(8, "press", 4242, reference=True)
    assert prompt_for(instrumented) is SYSTEM_PROMPT_WITH_REFERENCE
    assert "reference" in prompt_for(instrumented).lower()


def test_the_table_carries_all_three_columns():
    rows, _ = build_stream(6, "none", 31337)
    text = user_prompt(rows)
    assert text.count("bottom") == 6
    assert text.count("top") == 6
    assert text.count("tilt") == 6


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_produces_a_full_run(scenario):
    rows, truth = build_stream(24, scenario, 55555)
    assert truth == scenario
    assert len(rows) == 24
    assert all({"part", "seam_bottom_mm", "seam_top_mm", "tilt_mm"} <= set(r) for r in rows)


def test_generated_examples_support_their_own_labels(tmp_path):
    """Every written example has to be one a person could have labelled correctly."""
    out = tmp_path / "set"
    result = subprocess.run(
        [sys.executable, "scripts/make_drift_dataset.py",
         "--train", "30", "--val", "9", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr

    rows = [json.loads(line) for line in
            (out / "drift_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30

    seen = set()
    for row in rows:
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] == SYSTEM_PROMPT

        answer = json.loads(row["messages"][2]["content"])
        assert answer["cause"] in SCENARIOS
        assert 0.0 <= answer["confidence"] <= 1.0
        assert answer["evidence"]
        seen.add(answer["cause"])

    # Balanced, or the model learns the prior rather than the evidence.
    assert seen == set(SCENARIOS)


def test_training_and_validation_never_share_a_run(tmp_path):
    out = tmp_path / "set"
    subprocess.run(
        [sys.executable, "scripts/make_drift_dataset.py",
         "--train", "30", "--val", "9", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=300, check=True,
    )
    read = lambda name: {  # noqa: E731
        json.loads(line)["messages"][1]["content"]
        for line in (out / name).read_text(encoding="utf-8").splitlines()
    }
    assert not (read("drift_train.jsonl") & read("drift_val.jsonl"))
