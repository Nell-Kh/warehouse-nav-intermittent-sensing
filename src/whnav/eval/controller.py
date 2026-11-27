"""Shared interfaces for evaluation controllers and occupancy predictors."""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from whnav.env.grid import Cell, GridMap

METHODS = ("ppo4", "rppo", "rppo_aux", "planner", "lastseen", "privileged", "freeze")


@dataclass(frozen=True)
class StaticInfo:
    """Information available before choosing the first action.

    ``episode_seed`` is an optional seed for controller randomness. It is
    supplied by evaluation independently of the simulator's random state.
    """

    grid: GridMap
    start: Cell
    goal: Cell
    horizon: int
    episode_seed: int | None = None


class Controller(Protocol):
    """Choose one action per observation and manage memory within an episode.

    ``name`` is a method identifier from ``METHODS``. Only the privileged
    A* reference sets ``privileged`` to True.
    """

    name: str
    privileged: bool

    def reset(self, static: StaticInfo) -> None:
        """Start an episode and clear any memory from the previous one."""
        ...

    def act(self, obs: dict[str, np.ndarray], info: dict[str, Any] | None = None) -> int:
        """Choose a move from a raw observation, with extra info only for privileged A*."""
        ...


class OccupancyPredictor(Protocol):
    """Prediction interface for the motion planner and method 3."""

    def observe(self, obs: dict[str, np.ndarray]) -> None:
        """Update memory from one observation during a replay."""
        ...

    def last_occupancy_scores(self) -> np.ndarray:
        """Return the latest robot-centered 9×9 scores, with NaN off the map."""
        ...
