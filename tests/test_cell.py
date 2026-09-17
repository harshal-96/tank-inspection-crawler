"""Tests for the edge-side modules: geometry, defect injection and the service client.

No Isaac and no network. These cover the parts the simulation depends on, so a
mistake here is caught in a second rather than after an hour-long campaign.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from cell import defects, geometry
from cell.client import CellClient, replay_offline_queue

# ------------------------------------------------------------------------ geometry


def test_part_arriving_perfectly_sits_at_the_nominal_joint():
    assert geometry.true_joint(geometry.FIXTURE_X, 0.0, 0.0) == pytest.approx(
        geometry.nominal_joint()
    )


def test_lateral_offset_moves_the_joint_by_the_same_amount():
    nominal = geometry.nominal_joint()
    shifted = geometry.true_joint(geometry.FIXTURE_X, 0.030, 0.0)
    assert shifted[0] == pytest.approx(nominal[0])
    assert shifted[1] == pytest.approx(nominal[1] + 0.030)


def test_quarter_turn_rotates_the_joint_about_the_part_origin():
    local_x, local_y = geometry.JOINT_LOCAL
    x, y = geometry.true_joint(0.0, 0.0, math.pi / 2)
    assert x == pytest.approx(-local_y)
    assert y == pytest.approx(local_x)


def test_yaw_alone_moves_the_joint():
    """A part square on the stop but rotated still has a displaced joint."""
    straight = geometry.true_joint(geometry.FIXTURE_X, 0.0, 0.0)
    rotated = geometry.true_joint(geometry.FIXTURE_X, 0.0, math.radians(2.0))
    assert geometry.distance_mm(straight, rotated) > 5.0


def test_blind_error_is_zero_for_a_perfect_arrival():
    assert geometry.blind_error_mm(geometry.FIXTURE_X, 0.0, 0.0) == pytest.approx(0.0)


def test_blind_error_grows_with_offset():
    small = geometry.blind_error_mm(geometry.FIXTURE_X, 0.005, 0.0)
    large = geometry.blind_error_mm(geometry.FIXTURE_X, 0.050, 0.0)
    assert large > small
    assert large == pytest.approx(50.0)


def test_distance_mm_converts_metres():
    assert geometry.distance_mm((0.0, 0.0), (0.003, 0.004)) == pytest.approx(5.0)


def test_torch_target_stands_off_by_the_stickout():
    joint = geometry.nominal_joint()
    tip = geometry.torch_target(joint[0], joint[1], 0.0)
    assert geometry.distance_mm(joint, tip) == pytest.approx(geometry.STICKOUT_M * 1000.0)


# ------------------------------------------------------------------------- defects


def test_clean_bead_is_one_continuous_span():
    plan = defects.clean_bead()
    assert plan.clean
    assert plan.segments == [(0.0, 1.0)]
    assert not plan.blobs


def test_campaign_is_reproducible_from_the_seed():
    a = defects.plan_campaign(40, seed=11)
    b = defects.plan_campaign(40, seed=11)
    assert [p.defect for p in a] == [p.defect for p in b]


def test_different_seeds_differ():
    a = defects.plan_campaign(40, seed=1)
    b = defects.plan_campaign(40, seed=2)
    assert [p.defect for p in a] != [p.defect for p in b]


def test_defect_rate_is_respected():
    plans = defects.plan_campaign(400, seed=5, defect_rate=0.35)
    rate = sum(0 if p.clean else 1 for p in plans) / len(plans)
    assert 0.28 < rate < 0.42


def test_zero_defect_rate_gives_only_clean_parts():
    assert all(p.clean for p in defects.plan_campaign(50, seed=3, defect_rate=0.0))


def test_full_defect_rate_gives_no_clean_parts():
    assert not any(p.clean for p in defects.plan_campaign(50, seed=3, defect_rate=1.0))


def test_invalid_defect_rate_rejected():
    with pytest.raises(ValueError):
        defects.plan_bead(random.Random(0), defect_rate=1.5)


def test_restricting_kinds_only_produces_those_kinds():
    plans = defects.plan_campaign(60, seed=7, defect_rate=1.0, kinds=(defects.SKIP,))
    assert {p.defect for p in plans} == {defects.SKIP}


def test_skip_leaves_a_real_gap_in_the_bead():
    plans = defects.plan_campaign(60, seed=9, defect_rate=1.0, kinds=(defects.SKIP,))
    for plan in plans:
        assert len(plan.segments) == 2
        (_, first_end), (second_start, _) = plan.segments
        assert second_start > first_end  # there is bare joint between them


def test_every_segment_is_ordered_and_inside_the_joint():
    for plan in defects.plan_campaign(200, seed=13, defect_rate=0.8):
        for start, end in plan.segments:
            assert 0.0 <= start < end <= 1.0


def test_severity_is_normalised():
    for plan in defects.plan_campaign(200, seed=17, defect_rate=1.0):
        assert 0.0 <= plan.severity <= 1.0


def test_spatter_lands_beside_the_bead_not_on_it():
    plans = defects.plan_campaign(30, seed=21, defect_rate=1.0, kinds=(defects.SPATTER,))
    for plan in plans:
        assert all(abs(b.offset) > 0.010 for b in plan.blobs)


def test_porosity_sits_on_the_bead_face():
    plans = defects.plan_campaign(30, seed=23, defect_rate=1.0, kinds=(defects.POROSITY,))
    for plan in plans:
        assert plan.blobs
        assert all(abs(b.offset) <= 0.005 for b in plan.blobs)


def test_describe_never_leaks_for_clean_parts():
    assert defects.clean_bead().describe() == "clean bead"


# -------------------------------------------------------------------------- client


def test_offline_posts_are_queued_not_lost(tmp_path: Path):
    queue = tmp_path / "queue.jsonl"
    # Port 9 is the discard service, reliably closed.
    client = CellClient(
        base_url="http://127.0.0.1:9", fallback_path=queue, verbose=False, timeout=0.5
    )

    assert client.create_part(spawn_dy_mm=12.0) is None
    assert client.offline is True

    rows = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["path"] == "/api/parts"
    assert rows[0]["body"]["spawn_dy_mm"] == 12.0


def test_offline_queue_accumulates_across_calls(tmp_path: Path):
    queue = tmp_path / "queue.jsonl"
    client = CellClient(
        base_url="http://127.0.0.1:9", fallback_path=queue, verbose=False, timeout=0.5
    )

    client.create_part()
    client.post_measurement("P-000001", (4.4, -0.6), 0.9, truth=(4.4, -0.6))
    client.post_inspection("P-000001", "a.png", "pass")

    rows = queue.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3


def test_measurement_body_carries_truth_only_when_given(tmp_path: Path):
    queue = tmp_path / "queue.jsonl"
    client = CellClient(
        base_url="http://127.0.0.1:9", fallback_path=queue, verbose=False, timeout=0.5
    )

    client.post_measurement("P-1", (1.0, 2.0), 0.8)
    client.post_measurement("P-2", (1.0, 2.0), 0.8, truth=(1.1, 2.1))

    bodies = [json.loads(line)["body"] for line in queue.read_text().splitlines()]
    assert "true_x" not in bodies[0]
    assert bodies[1]["true_x"] == 1.1


def test_replay_of_a_missing_queue_is_a_noop(tmp_path: Path):
    assert replay_offline_queue(tmp_path / "nothing.jsonl") == (0, 0)


def test_replay_keeps_rows_that_still_fail(tmp_path: Path):
    queue = tmp_path / "queue.jsonl"
    queue.write_text(json.dumps({"path": "/api/parts", "body": {}}) + "\n", encoding="utf-8")

    sent, failed = replay_offline_queue(queue, base_url="http://127.0.0.1:9")
    assert (sent, failed) == (0, 1)
    assert queue.exists()  # nothing was thrown away


def test_healthy_is_false_when_service_is_down():
    assert CellClient(base_url="http://127.0.0.1:9", verbose=False, timeout=0.5).healthy() is False
