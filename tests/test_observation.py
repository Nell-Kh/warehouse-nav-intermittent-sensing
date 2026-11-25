import numpy as np
import pytest

from whnav.env.grid import GridMap
from whnav.env.observation import (
    Channel,
    Scalar,
    encode_observation,
    steps_since_valid,
    window_mask,
)

OPEN = GridMap.from_strings(["." * 12] * 12)
SHELVES = GridMap.from_strings(
    [
        ".........",
        ".##.##.#.",
        ".##.##.#.",
        ".........",
        ".##.##.#.",
        ".........",
    ]
)
UP = np.zeros(10, dtype=bool)  # sensor always up
DOWN_AT_0 = np.array([True] + [False] * 9)
ROBOT, GOAL = (5, 5), (11, 11)


def obs(grid=OPEN, robot=ROBOT, goal=GOAL, workers=(), schedule=UP, index=0):
    return encode_observation(grid, robot, goal, list(workers), schedule, index)


def worker_state(o, cell):
    """Which of the three worker channels is set at ``cell``."""
    g = o["grid"]
    set_channels = [
        ch for ch in (Channel.SEEN_EMPTY, Channel.SEEN_WORKER, Channel.NOT_OBSERVED) if g[ch][cell]
    ]
    assert len(set_channels) <= 1
    return set_channels[0] if set_channels else None


# --- shapes and static channels ---


def test_shapes_and_dtypes():
    o = obs(grid=SHELVES, robot=(0, 0), goal=(5, 8))
    assert o["grid"].shape == (6, 6, 9) and o["grid"].dtype == np.float32
    assert o["scalars"].shape == (3,) and o["scalars"].dtype == np.float32
    assert set(np.unique(o["grid"])) <= {0.0, 1.0}


def test_static_map_robot_and_goal_channels():
    o = obs(grid=SHELVES, robot=(3, 3), goal=(5, 8))
    g = o["grid"]
    assert np.array_equal(g[Channel.BLOCKED], SHELVES.blocked.astype(np.float32))
    assert g[Channel.ROBOT].sum() == 1 and g[Channel.ROBOT][3, 3] == 1
    assert g[Channel.GOAL].sum() == 1 and g[Channel.GOAL][5, 8] == 1


def test_goal_is_visible_even_outside_the_window_and_during_an_outage():
    o = obs(schedule=DOWN_AT_0)
    assert o["grid"][Channel.GOAL][GOAL] == 1


# --- the 7x7 window ---


def test_window_is_7x7_centered_on_robot():
    m = window_mask(OPEN, ROBOT)
    assert m.sum() == 49
    assert m[2, 2] and m[8, 8] and m[2, 8] and m[8, 2]  # corners, 3 cells away
    assert not m[1, 5] and not m[9, 5] and not m[5, 1] and not m[5, 9]  # 4 cells away


def test_window_is_clipped_at_the_map_edge():
    assert window_mask(OPEN, (0, 0)).sum() == 16
    o = obs(robot=(0, 0))
    assert o["grid"][Channel.SEEN_EMPTY].sum() == 16


def test_worker_inside_window_is_seen():
    o = obs(workers=[(2, 8)])  # window corner
    assert worker_state(o, (2, 8)) == Channel.SEEN_WORKER
    assert o["grid"][Channel.SEEN_WORKER].sum() == 1


def test_worker_outside_window_is_not_observed_never_empty():
    o = obs(workers=[(5, 9)])  # 4 cells east
    assert worker_state(o, (5, 9)) == Channel.NOT_OBSERVED
    assert o["grid"][Channel.SEEN_WORKER].sum() == 0


def test_workers_outside_window_do_not_change_the_observation():
    a = obs(workers=[(4, 4), (0, 0), (11, 3)])
    b = obs(workers=[(4, 4), (10, 10)])
    assert np.array_equal(a["grid"], b["grid"]) and np.array_equal(a["scalars"], b["scalars"])


def test_shelves_do_not_block_the_view():
    # both workers are inside the window with shelf cells between them and the robot
    o = obs(grid=SHELVES, robot=(3, 4), goal=(5, 8), workers=[(0, 3), (5, 1)])
    assert worker_state(o, (0, 3)) == Channel.SEEN_WORKER
    assert worker_state(o, (5, 1)) == Channel.SEEN_WORKER


# --- three states per free cell ---


@pytest.mark.parametrize("schedule", [UP, DOWN_AT_0])
def test_every_free_cell_has_exactly_one_worker_state_and_blocked_cells_none(schedule):
    o = obs(grid=SHELVES, robot=(3, 3), goal=(5, 8), workers=[(3, 4), (0, 8)], schedule=schedule)
    total = o["grid"][[Channel.SEEN_EMPTY, Channel.SEEN_WORKER, Channel.NOT_OBSERVED]].sum(axis=0)
    assert np.array_equal(total, (~SHELVES.blocked).astype(np.float32))


def test_workers_are_anonymous_and_may_share_a_cell():
    a = obs(workers=[(4, 4), (6, 6), (6, 6)])
    b = obs(workers=[(6, 6), (4, 4)])
    assert np.array_equal(a["grid"], b["grid"])
    assert a["grid"][Channel.SEEN_WORKER].sum() == 2


# --- outages ---


def test_outage_hides_the_whole_window():
    o = obs(workers=[(5, 6)], schedule=DOWN_AT_0)
    g = o["grid"]
    assert g[Channel.SEEN_EMPTY].sum() == 0 and g[Channel.SEEN_WORKER].sum() == 0
    assert g[Channel.NOT_OBSERVED].sum() == 144
    assert o["scalars"][Scalar.VALID] == 0.0


def test_outage_observation_does_not_depend_on_workers():
    a = obs(workers=[(5, 6), (4, 4)], schedule=DOWN_AT_0)
    b = obs(workers=[], schedule=DOWN_AT_0)
    assert np.array_equal(a["grid"], b["grid"]) and np.array_equal(a["scalars"], b["scalars"])


# --- scalars ---


def test_steps_since_valid():
    s = np.array([True, True, False, True, True, True, False])
    assert [steps_since_valid(s, i) for i in range(7)] == [1, 2, 0, 1, 2, 3, 0]
    with pytest.raises(ValueError):
        steps_since_valid(s, 7)


def test_scalars():
    s = np.array([False, True, True, False, False, False, False, False, False, False])
    o = obs(schedule=s, index=2)
    assert o["scalars"][Scalar.VALID] == 0.0
    assert o["scalars"][Scalar.SINCE_VALID] == pytest.approx(2 / 10)
    assert o["scalars"][Scalar.TIME_LEFT] == pytest.approx(8 / 10)
    first = obs(schedule=s, index=0)
    assert first["scalars"].tolist() == [1.0, 0.0, 1.0]


# --- input validation ---


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        obs(index=10)
    with pytest.raises(ValueError):
        obs(grid=SHELVES, robot=(1, 1), goal=(5, 8))
    with pytest.raises(ValueError):
        obs(grid=SHELVES, robot=(0, 0), goal=(1, 1))
    with pytest.raises(ValueError):
        obs(grid=SHELVES, robot=(0, 0), goal=(5, 8), workers=[(1, 1)])
