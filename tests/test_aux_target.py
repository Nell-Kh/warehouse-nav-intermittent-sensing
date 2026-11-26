"""Tests for auxiliary occupancy labels, masks, and Gymnasium integration."""

from contextlib import closing

import gymnasium as gym
import numpy as np
import pytest

from whnav.agents.aux_target import AUX_SIZE, AuxTargetWrapper, aux_target
from whnav.env.grid import Action, GridMap
from whnav.env.observation import Channel, encode_observation
from whnav.env.scenario import make_scenario
from whnav.env.warehouse_env import WarehouseEnv
from whnav.env.workers import Heading, WorkerState


def test_no_outage_masks_only_the_outer_ring() -> None:
    grid = GridMap.from_strings(["." * 11] * 11)
    workers = [(5, 9), (5, 9), (2, 7), (9, 9), (0, 5)]

    labels, mask = aux_target(grid, (5, 5), workers, outage=False)

    expected_mask = np.ones((9, 9), dtype=bool)
    expected_mask[1:-1, 1:-1] = False
    expected_labels = np.zeros((9, 9), dtype=np.float32)
    expected_labels[4, 8] = expected_labels[8, 8] = 1.0
    assert AUX_SIZE == 9
    assert labels.dtype == np.float32
    assert mask.dtype == np.bool_
    assert mask.sum() == 32
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(labels, expected_labels)


def test_outage_masks_every_free_cell_except_robot() -> None:
    blocked = np.zeros((11, 11), dtype=bool)
    blocked[1, 1] = blocked[5, 6] = True
    workers = [(5, 5), (5, 6), (1, 1), (4, 6), (5, 10)]

    labels, mask = aux_target(GridMap(blocked), (5, 5), workers, outage=True)

    expected_mask = ~blocked[1:10, 1:10]
    expected_mask[4, 4] = False
    expected_labels = np.zeros((9, 9), dtype=np.float32)
    expected_labels[3, 5] = 1.0
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(labels, expected_labels)


@pytest.mark.parametrize("outage", [False, True])
def test_map_edge_clips_targets_and_excludes_shelves(outage: bool) -> None:
    grid = GridMap.from_strings(["......", "#.....", "......", "......", "....#."])
    workers = [(0, 4), (4, 0), (4, 4), (1, 0), (-1, 0), (0, -1), (0, 5), (2, 3)]

    labels, mask = aux_target(grid, (0, 0), workers, outage=outage)

    expected_mask = np.zeros((9, 9), dtype=bool)
    expected_mask[4:, 8] = expected_mask[8, 4:] = True
    if outage:
        expected_mask[4:, 4:] = True
        expected_mask[5, 4] = False
    expected_mask[4, 4] = False
    expected_mask[8, 8] = False
    expected_labels = np.zeros((9, 9), dtype=np.float32)
    expected_labels[4, 8] = expected_labels[8, 4] = 1.0
    expected_labels[6, 7] = float(outage)
    assert not labels[~mask].any()
    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(labels, expected_labels)


@pytest.mark.parametrize("seed", [0, 7, 42, 101])
@pytest.mark.parametrize("outage", [False, True])
def test_mask_matches_the_unobserved_channel(seed: int, outage: bool) -> None:
    rng = np.random.default_rng(seed)
    height, width = 11, 13
    blocked = rng.random((height, width)) < 0.25
    robot = (int(rng.integers(height)), int(rng.integers(width)))
    goal = (robot[0], (robot[1] + 1) % width)
    blocked[robot] = blocked[goal] = False
    free_cells = np.argwhere(~blocked)
    worker_cells = free_cells[rng.choice(len(free_cells), size=6, replace=False)]
    workers = [(int(row), int(col)) for row, col in worker_cells]
    grid = GridMap(blocked)
    schedule = np.array([not outage, outage], dtype=bool)

    observation = encode_observation(grid, robot, goal, workers, schedule, index=1)
    _, mask = aux_target(grid, robot, workers, outage)

    radius = AUX_SIZE // 2
    unobserved = observation["grid"][Channel.NOT_OBSERVED].astype(bool)
    padded = np.pad(unobserved, radius)
    row, col = robot
    expected = padded[row : row + AUX_SIZE, col : col + AUX_SIZE]
    expected[radius, radius] = False
    np.testing.assert_array_equal(mask, expected)


def test_no_outage_can_have_an_empty_mask() -> None:
    grid = GridMap.from_strings(["..."] * 3)

    labels, mask = aux_target(grid, (1, 1), [(0, 0)], outage=False)

    assert not mask.any()
    assert not labels.any()


def test_wrapper_refreshes_the_map_on_reset() -> None:
    layout = ("......",) * 5
    original = make_scenario(layout, (0, 0), (0, 1))
    override = make_scenario(("....#.", *layout[1:]), (0, 0), (0, 1))

    with closing(AuxTargetWrapper(WarehouseEnv([original]))) as env:
        observation, original_info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert original_info["aux_mask"][4, 8]

        observation, override_info = env.reset(options={"scenario": override})
        assert observation.keys() == {"grid", "scalars"}
        assert not override_info["aux_mask"][4, 8]
        assert override_info["aux_mask"][8, 4]
        assert original_info["aux_mask"][4, 8]


def test_wrapper_preserves_step_and_reset_targets_with_same_step_autoreset() -> None:
    layout = ("." * 9,) * 7
    workers = [WorkerState((2, 5), Heading.W)]
    terminal_case = make_scenario(
        layout, (2, 2), (2, 3), workers, pattern="scattered", f=0.5, schedule=[False, True]
    )
    running_case = make_scenario(
        layout,
        (2, 2),
        (2, 7),
        workers,
        pattern="scattered",
        f=0.1,
        schedule=[False, True] + [False] * 8,
    )
    with closing(
        gym.vector.SyncVectorEnv(
            [
                lambda: AuxTargetWrapper(WarehouseEnv([terminal_case])),
                lambda: AuxTargetWrapper(WarehouseEnv([running_case])),
            ],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    ) as envs:
        observation, reset_info = envs.reset(seed=0)
        assert observation.keys() == {"grid", "scalars"}
        assert not reset_info["aux_labels"].any()

        observation, _, terminated, truncated, info = envs.step(np.array([Action.E, Action.WAIT]))
        assert observation.keys() == {"grid", "scalars"}
        np.testing.assert_array_equal(terminated, [True, False])
        assert not truncated.any()
        np.testing.assert_array_equal(info["t"], [0, 1])
        np.testing.assert_array_equal(info["_final_info"], [True, False])

        final_info = info["final_info"]
        for batch in (reset_info, info, final_info):
            assert batch["aux_labels"].shape == batch["aux_mask"].shape == (2, 9, 9)
            assert batch["aux_labels"].dtype == np.float32
            assert batch["aux_mask"].dtype == np.bool_
            assert not batch["aux_mask"][:, 4, 4].any()
            assert not batch["aux_labels"][:, 4, 4].any()

        # The terminal environment's reset and final info belong to different observations.
        np.testing.assert_array_equal(info["aux_labels"][0], reset_info["aux_labels"][0])
        np.testing.assert_array_equal(info["aux_mask"][0], reset_info["aux_mask"][0])
        assert final_info["outcome"][0] == "success"
        assert final_info["aux_labels"][0, 4, 5] == 1.0
        assert info["aux_labels"][1, 4, 6] == 1.0
        assert info["aux_labels"][1, 4, 7] == 0.0
