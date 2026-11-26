"""Auxiliary occupancy targets for unobserved warehouse cells (SPEC §4)."""

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from whnav.env.grid import Cell, GridMap
from whnav.env.observation import WINDOW_RADIUS, Channel

AUX_SIZE = 9


def aux_target(
    grid: GridMap,
    robot: Cell,
    worker_positions: Sequence[Cell],
    outage: bool,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Return occupancy labels and a validity mask for the robot-centered 9×9 area.

    Coordinates follow the map's (row, column) convention. The robot's own
    cell at index (4, 4) is always excluded from the mask. The boolean mask
    covers traversable, in-map cells outside the current 7×7 sensor window.
    During an outage it covers every other traversable, in-map cell in the
    area. Earlier readings do not affect the mask.

    The float32 labels are 1.0 at masked cells with one or more workers and 0.0
    elsewhere. Worker positions are privileged ground truth for supervision and
    prediction-quality evaluation; they are never supplied to the policy.
    """
    radius = AUX_SIZE // 2
    labels = np.zeros((AUX_SIZE, AUX_SIZE), dtype=np.float32)
    mask = np.zeros((AUX_SIZE, AUX_SIZE), dtype=bool)
    occupied = set(worker_positions)
    for row in range(AUX_SIZE):
        dr = row - radius
        for col in range(AUX_SIZE):
            dc = col - radius
            distance = max(abs(dr), abs(dc))
            if distance == 0 or (not outage and distance <= WINDOW_RADIUS):
                continue
            cell = (robot[0] + dr, robot[1] + dc)
            if grid.is_free(cell):
                mask[row, col] = True
                labels[row, col] = cell in occupied
    return labels, mask


class AuxTargetWrapper(gym.Wrapper):
    """Add occupancy supervision to WarehouseEnv info on reset and step."""

    _grid: GridMap

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray[np.float32]], dict[str, Any]]:
        """Cache the new episode's static map and label its first observation."""
        observation, info = super().reset(seed=seed, options=options)
        self._grid = GridMap(observation["grid"][Channel.BLOCKED].astype(bool))
        return observation, self._add_targets(info)

    def step(
        self, action: int
    ) -> tuple[dict[str, NDArray[np.float32]], float, bool, bool, dict[str, Any]]:
        """Attach labels to the info corresponding to the observation returned by step."""
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated, truncated, self._add_targets(info)

    def _add_targets(self, info: dict[str, Any]) -> dict[str, Any]:
        labels, mask = aux_target(
            self._grid, info["robot"], info["worker_positions"], info["outage"]
        )
        return {**info, "aux_labels": labels, "aux_mask": mask}
