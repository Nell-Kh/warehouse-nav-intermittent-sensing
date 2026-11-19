from collections import Counter

import numpy as np
import pytest

from whnav.env.grid import GridMap
from whnav.env.workers import (
    HEADING_DELTAS,
    Heading,
    WorkerState,
    is_decision_point,
    step_worker,
    step_workers,
)

N, E, S, W = Heading.N, Heading.E, Heading.S, Heading.W

CORRIDOR = GridMap.from_strings(
    [
        "#####",
        "#...#",
        "#####",
    ]
)
CORNER = GridMap.from_strings(
    [
        "####",
        "#..#",
        "##.#",
        "####",
    ]
)
PLUS = GridMap.from_strings(
    [
        "#####",
        "##.##",
        "#...#",
        "##.##",
        "#####",
    ]
)
T_NO_STRAIGHT = GridMap.from_strings(
    [
        "#####",
        "#####",
        "#...#",
        "##.##",
        "#####",
    ]
)
L_LEFT_ONLY = GridMap.from_strings(
    [
        "#####",
        "#####",
        "#..##",
        "##.##",
        "#####",
    ]
)
WAREHOUSE = GridMap.from_strings(
    [
        "..........",
        ".##.##.##.",
        ".##.##.##.",
        "..........",
        ".##.##.##.",
        "..........",
    ]
)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_heading_turns():
    assert (N.right, N.left, N.back) == (E, W, S)
    assert (W.right, W.left, W.back) == (N, S, E)


# --- aisles: always forward, no randomness ---


def test_aisle_moves_forward_without_using_randomness():
    g = rng()
    before = g.bit_generator.state
    state = WorkerState((1, 1), E)
    assert not is_decision_point(CORRIDOR, state)
    assert step_worker(CORRIDOR, state, p=0.0, rng=g) == WorkerState((1, 2), E)
    assert g.bit_generator.state == before


# --- decision points ---


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_dead_end_turns_back(p):
    assert step_worker(CORRIDOR, WorkerState((1, 3), E), p, rng()) == WorkerState((1, 2), W)


@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
def test_corner_takes_the_only_turn(p):
    for seed in range(20):
        nxt = step_worker(CORNER, WorkerState((1, 2), E), p, rng(seed))
        assert nxt == WorkerState((2, 2), S)


def test_rule_goes_straight_when_possible():
    assert step_worker(PLUS, WorkerState((2, 2), N), 1.0, rng()).heading == N


def test_rule_prefers_right_over_left():
    assert step_worker(T_NO_STRAIGHT, WorkerState((2, 2), N), 1.0, rng()).heading == E


def test_rule_takes_left_when_straight_and_right_blocked():
    assert step_worker(L_LEFT_ONLY, WorkerState((2, 2), N), 1.0, rng()).heading == W


def heading_frequencies(grid, state, p, n=6000, seed=1):
    g = rng(seed)
    counts = Counter(step_worker(grid, state, p, g).heading for _ in range(n))
    return {h: counts[h] / n for h in Heading}


def test_random_choice_is_uniform_and_never_turns_back():
    freq = heading_frequencies(PLUS, WorkerState((2, 2), N), p=0.0)
    assert freq[S] == 0.0
    for h in (N, E, W):
        assert freq[h] == pytest.approx(1 / 3, abs=0.03)


def test_rule_adherence_mixes_rule_and_random():
    # straight is the rule's choice: P(N) = p + (1 - p) / 3
    p = 0.5
    freq = heading_frequencies(PLUS, WorkerState((2, 2), N), p=p)
    assert freq[N] == pytest.approx(p + (1 - p) / 3, abs=0.03)
    assert freq[E] == pytest.approx((1 - p) / 3, abs=0.03)
    assert freq[W] == pytest.approx((1 - p) / 3, abs=0.03)
    assert freq[S] == 0.0


# --- whole trajectories ---


def run(seed, steps=300, p=0.3):
    starts = [WorkerState((0, 0), E), WorkerState((3, 9), W), WorkerState((5, 3), N)]
    g = rng(seed)
    traj = [starts]
    for _ in range(steps):
        traj.append(step_workers(WAREHOUSE, traj[-1], p, g))
    return traj


def test_every_move_is_one_step_onto_a_free_cell():
    traj = run(seed=3)
    for before, after in zip(traj, traj[1:], strict=False):
        for b, a in zip(before, after, strict=True):
            assert WAREHOUSE.is_free(a.pos)
            dr, dc = HEADING_DELTAS[a.heading]
            assert a.pos == (b.pos[0] + dr, b.pos[1] + dc)


def test_same_seed_replays_identically():
    assert run(seed=7) == run(seed=7)


def test_different_seeds_give_different_trajectories():
    assert run(seed=7) != run(seed=8)


def test_workers_may_share_a_cell():
    states = [WorkerState((1, 1), E), WorkerState((1, 1), E)]
    after = step_workers(CORRIDOR, states, 0.5, rng())
    assert after[0].pos == after[1].pos == (1, 2)


# --- input validation ---


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_invalid_p_raises(p):
    with pytest.raises(ValueError):
        step_worker(CORRIDOR, WorkerState((1, 1), E), p, rng())


def test_worker_on_blocked_cell_raises():
    with pytest.raises(ValueError):
        step_worker(CORRIDOR, WorkerState((0, 0), E), 0.5, rng())


def test_enclosed_worker_raises():
    g = GridMap.from_strings(["###", "#.#", "###"])
    with pytest.raises(ValueError):
        step_worker(g, WorkerState((1, 1), N), 0.5, rng())
