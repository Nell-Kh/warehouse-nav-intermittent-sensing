"""Static warehouse map, robot actions and robot movement (SPEC §2.1, §2.2, §2.4).

Coordinates are ``(row, col)`` with row 0 at the top. A cell is either free or
blocked (shelf or outer wall). The map never changes during an episode.
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum

import numpy as np

Cell = tuple[int, int]


class Action(IntEnum):
    """The robot's five actions (SPEC §2.2)."""

    N = 0
    S = 1
    E = 2
    W = 3
    WAIT = 4


ACTION_DELTAS: dict[Action, Cell] = {
    Action.N: (-1, 0),
    Action.S: (1, 0),
    Action.E: (0, 1),
    Action.W: (0, -1),
    Action.WAIT: (0, 0),
}

FREE_CHAR = "."
BLOCKED_CHAR = "#"


class GridMap:
    """Immutable static map: ``blocked[r, c]`` is True for shelves and walls."""

    def __init__(self, blocked: np.ndarray) -> None:
        arr = np.asarray(blocked)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"blocked must be a non-empty 2D array, got shape {arr.shape}")
        if arr.dtype != np.bool_:
            raise TypeError(f"blocked must be a boolean array, got dtype {arr.dtype}")
        self._blocked = arr.copy()
        self._blocked.setflags(write=False)

    @classmethod
    def from_strings(cls, rows: list[str]) -> GridMap:
        """Build a map from text rows, using ``#`` for blocked and ``.`` for free cells."""
        if not rows:
            raise ValueError("rows must not be empty")
        width = len(rows[0])
        for i, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"row {i} has length {len(row)}, expected {width}")
            bad = set(row) - {FREE_CHAR, BLOCKED_CHAR}
            if bad:
                raise ValueError(f"row {i} has invalid characters {sorted(bad)}")
        blocked = np.array([[ch == BLOCKED_CHAR for ch in row] for row in rows], dtype=bool)
        return cls(blocked)

    @property
    def blocked(self) -> np.ndarray:
        """Read-only boolean array of shape ``(height, width)``."""
        return self._blocked

    @property
    def height(self) -> int:
        return self._blocked.shape[0]

    @property
    def width(self) -> int:
        return self._blocked.shape[1]

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Cell) -> bool:
        """True if the cell is inside the map and not a shelf or wall."""
        return self.in_bounds(cell) and not self._blocked[cell]

    def free_neighbors(self, cell: Cell) -> list[Cell]:
        """Free cells reachable from ``cell`` in one N/S/E/W move."""
        r, c = cell
        out = []
        for action in (Action.N, Action.S, Action.E, Action.W):
            dr, dc = ACTION_DELTAS[action]
            nxt = (r + dr, c + dc)
            if self.is_free(nxt):
                out.append(nxt)
        return out

    def __repr__(self) -> str:
        return f"GridMap(height={self.height}, width={self.width})"


def move_robot(grid: GridMap, pos: Cell, action: Action) -> Cell:
    """Return the robot's cell after ``action``.

    Moving into a shelf, a wall or off the map leaves the robot in place
    (week-1 decision, SPEC §2.2). The robot moves at most one cell per step.
    """
    if not grid.is_free(pos):
        raise ValueError(f"robot position {pos} is not a free cell")
    dr, dc = ACTION_DELTAS[Action(action)]
    target = (pos[0] + dr, pos[1] + dc)
    return target if grid.is_free(target) else pos


def shortest_path_length(grid: GridMap, start: Cell, goal: Cell) -> int | None:
    """BFS distance from ``start`` to ``goal`` on the static map, or None if unreachable."""
    for name, cell in (("start", start), ("goal", goal)):
        if not grid.is_free(cell):
            raise ValueError(f"{name} {cell} is not a free cell")
    if start == goal:
        return 0
    dist = {start: 0}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for nxt in grid.free_neighbors(cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                if nxt == goal:
                    return dist[nxt]
                queue.append(nxt)
    return None


def deadline(shortest_path: int) -> int:
    """Episode deadline ``T = 2L`` (SPEC §2.4). ``L`` must be at least 1."""
    if shortest_path < 1:
        raise ValueError(f"shortest path length must be >= 1, got {shortest_path}")
    return 2 * shortest_path
