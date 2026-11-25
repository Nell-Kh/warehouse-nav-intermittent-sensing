"""Observation encoding (SPEC §2.6). Every method gets exactly this input.

An observation is a dict with two float32 arrays:

``grid``, shape ``(6, height, width)``, one channel per :class:`Channel`:
    BLOCKED, ROBOT and GOAL are known globally and never hidden.
    SEEN_EMPTY, SEEN_WORKER and NOT_OBSERVED describe workers. Every free cell
    has exactly one of the three set; blocked cells have none. A cell is seen
    only if it lies in the 7x7 window around the robot and the sensor is up.
    The window is not blocked by shelves and is clipped at the map edge.

``scalars``, shape ``(3,)``, one entry per :class:`Scalar`:
    VALID is 1 when the sensor reading is valid and 0 during an outage.
    SINCE_VALID is the number of steps since the last valid reading, over T.
    TIME_LEFT is ``steps_left / T``.

Workers are anonymous: the encoding depends only on which cells are occupied.
A missing reading is never encoded as an empty cell.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from whnav.env.grid import Cell, GridMap

WINDOW_SIZE = 7
WINDOW_RADIUS = WINDOW_SIZE // 2


class Channel(IntEnum):
    BLOCKED = 0
    ROBOT = 1
    GOAL = 2
    SEEN_EMPTY = 3
    SEEN_WORKER = 4
    NOT_OBSERVED = 5


class Scalar(IntEnum):
    VALID = 0
    SINCE_VALID = 1
    TIME_LEFT = 2


def window_mask(grid: GridMap, robot: Cell, radius: int = WINDOW_RADIUS) -> np.ndarray:
    """Boolean ``(height, width)`` array: True for cells in the window around ``robot``."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    r, c = robot
    mask[max(r - radius, 0) : r + radius + 1, max(c - radius, 0) : c + radius + 1] = True
    return mask


def steps_since_valid(schedule: np.ndarray, index: int) -> int:
    """Steps since the last valid reading, at observation ``index`` of an outage schedule.

    0 when the current reading is valid. If no reading has been valid yet, the
    count runs from the start of the episode: ``index + 1``.
    """
    if not 0 <= index < len(schedule):
        raise ValueError(f"index {index} outside 0..{len(schedule) - 1}")
    valid = np.flatnonzero(~np.asarray(schedule[: index + 1], dtype=bool))
    last_valid = int(valid[-1]) if valid.size else -1
    return index - last_valid


def encode_observation(
    grid: GridMap,
    robot: Cell,
    goal: Cell,
    worker_positions: Sequence[Cell],
    schedule: np.ndarray,
    index: int,
) -> dict[str, np.ndarray]:
    """Observation the robot uses for its ``(index + 1)``-th action.

    ``schedule`` is the episode's outage schedule (length T, True = missing) and
    ``worker_positions`` are the true current positions. Workers outside the
    window, and all workers during an outage, have no effect on the result.
    """
    horizon = len(schedule)
    if not 0 <= index < horizon:
        raise ValueError(f"index {index} outside 0..{horizon - 1}")
    for name, cell in (("robot", robot), ("goal", goal)):
        if not grid.is_free(cell):
            raise ValueError(f"{name} {cell} is not a free cell")
    for cell in worker_positions:
        if not grid.is_free(cell):
            raise ValueError(f"worker position {cell} is not a free cell")

    outage = bool(schedule[index])
    free = ~grid.blocked
    seen = free & window_mask(grid, robot) if not outage else np.zeros_like(free)

    occupied = np.zeros_like(free)
    for cell in worker_positions:
        occupied[cell] = True

    planes = np.zeros((len(Channel), grid.height, grid.width), dtype=np.float32)
    planes[Channel.BLOCKED] = grid.blocked
    planes[Channel.ROBOT][robot] = 1.0
    planes[Channel.GOAL][goal] = 1.0
    planes[Channel.SEEN_EMPTY] = seen & ~occupied
    planes[Channel.SEEN_WORKER] = seen & occupied
    planes[Channel.NOT_OBSERVED] = free & ~seen

    scalars = np.zeros(len(Scalar), dtype=np.float32)
    scalars[Scalar.VALID] = 0.0 if outage else 1.0
    scalars[Scalar.SINCE_VALID] = steps_since_valid(schedule, index) / horizon
    scalars[Scalar.TIME_LEFT] = (horizon - index) / horizon
    return {"grid": planes, "scalars": scalars}
