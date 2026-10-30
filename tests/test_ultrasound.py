"""Tests for the simulated thickness gauge in tank/ultrasound.py.

The gauge is only worth showing if it reads the wall from the echoes and not from the
number it was handed, so these pin the physics: the echo spacing is the round trip at the
speed of sound, the reading follows the wall, and a plate whose velocity the gauge was not
calibrated for reads off by about the right amount.
"""

from __future__ import annotations

import pytest

from tank import ultrasound as us


@pytest.fixture(scope="module")
def gauge():
    return us.Gauge()


def test_calibration_lands_on_steel(gauge):
    # two blocks fix velocity and zero; numerical dispersion moves it by well under a percent
    assert abs(gauge.velocity - us.C_STEEL) / us.C_STEEL < 0.002


@pytest.mark.parametrize("thick", [3.86, 6.0, 10.0, 12.0])
def test_reading_follows_the_wall(gauge, thick):
    got, _, _ = gauge.read(thick)
    assert got == pytest.approx(thick, abs=0.02)


def test_echo_gap_is_the_round_trip(gauge):
    ts, x = gauge.listen(10.0)
    assert us.round_trip(ts, x) == pytest.approx(2 * 10e-3 / us.C_STEEL, rel=0.003)


def test_uncalibrated_velocity_shows_up_as_error(gauge):
    # steel half a percent faster than the blocks returns the echo sooner, so a gauge that
    # assumes the block velocity reads a 10 mm wall about 0.05 mm thin
    got, _, _ = gauge.read(10.0, c_steel=us.C_STEEL * 1.005)
    assert got - 10.0 == pytest.approx(-0.05, abs=0.015)


def test_a_lining_rings_louder_than_the_back_wall(gauge):
    # timed over the whole trace, the gauge locks onto the lining's own ringing and reads the
    # same wrong number whatever the wall; the window opened past the ringing reads the steel
    start, trips = us.past_the_lining(0.5)
    assert trips == 4
    assert start == pytest.approx(us.BLANK_S + 4 * 2 * 0.5e-3 / us.C_LINING)
    whole, _, _ = gauge.read(6.0, lining_mm=0.5)
    gated, _, _ = gauge.read(6.0, lining_mm=0.5, start_s=start)
    assert abs(whole - 6.0) > 1.0
    assert gated == pytest.approx(6.0, abs=0.02)


def test_first_echo_counts_the_lining_as_steel():
    # 0.5 mm of epoxy at 2540 m/s takes as long as 1.17 mm of steel
    assert us.first_echo_mm(3.86, 0.5) - 3.86 == pytest.approx(1.165, abs=0.001)


def test_grid_error_is_half_a_cell():
    # raw, the wall reads long by about half a cell; halving the grid halves it, and the
    # calibration's zero term is what takes it out
    rows = us.convergence()
    for r in rows:
        assert r["wall_long_by_um"] == pytest.approx(r["half_cell_um"], rel=0.15)
    assert rows[0]["wall_long_by_um"] / rows[-1]["wall_long_by_um"] == pytest.approx(4, rel=0.15)
