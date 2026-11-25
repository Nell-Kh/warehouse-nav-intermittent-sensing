from collections import Counter
from itertools import combinations

import numpy as np
import pytest

from whnav.env.outages import (
    block_count_range,
    consecutive_schedule,
    draw_block_count,
    expected_blocks,
    is_feasible,
    outage_runs,
    outage_schedule,
    realized_fraction,
    scattered_schedule,
)


def rng(seed=0):
    return np.random.default_rng(seed)


# --- counts and rounding ---


def test_block_count_range():
    assert block_count_range(10, 0.25) == (2, 3)  # 2.5 blocks expected
    assert block_count_range(30, 0.1) == (3, 3)  # 0.1 * 30 is 3.0000000000000004 in floats
    assert block_count_range(40, 0.2, d=4) == (2, 2)
    assert block_count_range(40, 0.0, d=4) == (0, 0)


def test_whole_block_count_uses_no_randomness():
    g = rng()
    before = g.bit_generator.state
    assert draw_block_count(40, 0.2, 4, g) == 2
    assert g.bit_generator.state == before


@pytest.mark.parametrize(("horizon", "f", "d"), [(30, 0.2, 4), (20, 0.3, 4), (28, 0.3, 8)])
def test_random_rounding_makes_expected_missing_fraction_exactly_f(horizon, f, d):
    g = rng(3)
    n = 20000
    draws = [draw_block_count(horizon, f, d, g) for _ in range(n)]
    low, high = block_count_range(horizon, f, d)
    assert set(draws) == {low, high}
    assert np.mean(draws) == pytest.approx(expected_blocks(horizon, f, d), abs=0.02)
    assert np.mean(draws) * d / horizon == pytest.approx(f, abs=0.01)


# --- scattered ---


@pytest.mark.parametrize("seed", range(25))
def test_scattered_has_exact_count_and_no_adjacent_drops(seed):
    s = scattered_schedule(40, 0.2, rng(seed))
    assert s.dtype == bool and s.shape == (40,)
    assert s.sum() == 8
    assert all(length == 1 for _, length in outage_runs(s))


def test_scattered_tightest_feasible_case_alternates():
    # 5 isolated drops in 9 steps: only one schedule exists
    s = scattered_schedule(9, 5 / 9, rng())
    assert s.tolist() == [True, False] * 4 + [True]


# --- consecutive ---


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("d", [2, 4, 8])
def test_consecutive_blocks_have_length_d_and_do_not_touch(seed, d):
    horizon, f = 80, 0.2
    s = consecutive_schedule(horizon, f, d, rng(seed))
    runs = outage_runs(s)
    assert len(runs) in block_count_range(horizon, f, d)
    assert all(length == d for _, length in runs)  # also proves blocks never merge
    assert s.sum() == d * len(runs)


def test_same_missing_fraction_when_d_divides_the_drop_count():
    scattered = scattered_schedule(80, 0.2, rng(1))
    consecutive = consecutive_schedule(80, 0.2, 4, rng(1))
    assert realized_fraction(scattered) == realized_fraction(consecutive) == pytest.approx(0.2)


def test_each_schedule_misses_by_less_than_d_steps():
    # 0.2 * 30 / 4 = 1.5 blocks: each schedule has 4 or 8 missing steps, 6 on average
    for seed in range(50):
        s = consecutive_schedule(30, 0.2, 4, rng(seed))
        assert s.sum() in (4, 8)


# --- uniform placement ---


def all_valid_schedules(horizon, n, d):
    slots = horizon - n * d + 1
    out = []
    for combo in combinations(range(slots), n):
        s = [False] * horizon
        for i, c in enumerate(combo):
            for k in range(d):
                s[c + i * d + k] = True
        out.append(tuple(s))
    return out


@pytest.mark.parametrize(("horizon", "f", "d"), [(6, 2 / 6, 1), (7, 4 / 7, 2), (9, 6 / 9, 3)])
def test_every_valid_schedule_is_equally_likely(horizon, f, d):
    n, high = block_count_range(horizon, f, d)
    assert n == high  # whole block count, so every schedule has n blocks
    valid = all_valid_schedules(horizon, n, d)
    g = rng(5)
    draws = 400 * len(valid)
    counts = Counter(tuple(outage_schedule(horizon, f, d, g).tolist()) for _ in range(draws))
    assert set(counts) == set(valid)
    for schedule in valid:
        assert counts[schedule] / draws == pytest.approx(1 / len(valid), rel=0.25)


# --- reproducibility and randomness use ---


def test_same_seed_gives_same_schedule():
    a = consecutive_schedule(60, 0.2, 3, rng(11))
    b = consecutive_schedule(60, 0.2, 3, rng(11))
    assert np.array_equal(a, b)


def test_no_outage_schedule_uses_no_randomness():
    g = rng()
    before = g.bit_generator.state
    s = scattered_schedule(20, 0.0, g)
    assert not s.any()
    assert g.bit_generator.state == before


# --- feasibility (SPEC §2.7) ---


def test_feasibility_matches_spec_bounds():
    for horizon in range(1, 41):
        for d in (1, 2, 3, 5):
            for f in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                low, high = block_count_range(horizon, f, d)
                expected = f == 0.0 or (low >= 1 and high * (d + 1) <= horizon + 1)
                assert is_feasible(horizon, f, d) == expected
                if expected:
                    s = outage_schedule(horizon, f, d, rng(horizon))
                    runs = outage_runs(s)
                    assert low <= len(runs) <= high
                    assert all(length == d for _, length in runs)
                else:
                    with pytest.raises(ValueError):
                        outage_schedule(horizon, f, d, rng(horizon))


def test_scattered_above_half_is_infeasible():
    assert is_feasible(10, 0.5)  # 5 drops, 5 <= floor(11 / 2)
    assert not is_feasible(10, 0.6)
    with pytest.raises(ValueError):
        scattered_schedule(10, 0.6, rng())


def test_fewer_than_one_expected_block_is_rejected():
    # 0.1 * 20 / 8 = 0.25 blocks: most episodes would silently have no outage
    assert not is_feasible(20, 0.1, 8)
    with pytest.raises(ValueError):
        consecutive_schedule(20, 0.1, 8, rng())


@pytest.mark.parametrize(
    ("horizon", "f", "d"), [(0, 0.2, 1), (10, -0.1, 1), (10, 1.0, 1), (10, 0.2, 0)]
)
def test_invalid_inputs_raise(horizon, f, d):
    with pytest.raises(ValueError):
        outage_schedule(horizon, f, d, rng())


# --- helpers ---


def test_outage_runs():
    s = np.array([True, False, True, True, False, False, True], dtype=bool)
    assert outage_runs(s) == [(0, 1), (2, 2), (6, 1)]
    assert outage_runs(np.zeros(4, dtype=bool)) == []


def test_realized_fraction_rejects_empty():
    with pytest.raises(ValueError):
        realized_fraction(np.zeros(0, dtype=bool))
