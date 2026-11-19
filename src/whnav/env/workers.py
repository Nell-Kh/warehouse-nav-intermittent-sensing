"""Worker movement (SPEC §2.3).

Each step, every worker moves exactly one cell (speed 1, week-1 decision):
- In an aisle, where the only way on besides turning back is straight ahead,
  the worker keeps going straight and uses no randomness.
- Anywhere else (intersection, corner, dead end) is a decision point. With
  probability ``p`` (rule adherence) the worker takes the first legal move in
  the order straight -> right -> left -> back. Otherwise it picks uniformly
  among the legal moves except back, and turns back only at a dead end.

Workers ignore the robot and each other, so their trajectories depend only on
the map, their start states, ``p`` and the random generator. Use a generator
dedicated to workers (seeded by the scenario's worker-trajectory seed) so the
trajectories are identical for every controller (SPEC §2.8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from whnav.env.grid import Cell, GridMap


class Heading(IntEnum):
    """Compass headings in clockwise order, so a right turn is ``+1`` mod 4."""

    N = 0
    E = 1
    S = 2
    W = 3

    @property
    def right(self) -> Heading:
        return Heading((self + 1) % 4)

    @property
    def back(self) -> Heading:
        return Heading((self + 2) % 4)

    @property
    def left(self) -> Heading:
        return Heading((self + 3) % 4)


HEADING_DELTAS: dict[Heading, Cell] = {
    Heading.N: (-1, 0),
    Heading.E: (0, 1),
    Heading.S: (1, 0),
    Heading.W: (0, -1),
}


@dataclass(frozen=True)
class WorkerState:
    """A worker's cell and the direction it last moved in."""

    pos: Cell
    heading: Heading


def neighbor(pos: Cell, heading: Heading) -> Cell:
    dr, dc = HEADING_DELTAS[heading]
    return (pos[0] + dr, pos[1] + dc)


def legal_headings(grid: GridMap, pos: Cell) -> list[Heading]:
    """Headings that lead from ``pos`` to a free cell, in N, E, S, W order."""
    return [h for h in Heading if grid.is_free(neighbor(pos, h))]


def forward_options(grid: GridMap, state: WorkerState) -> list[Heading]:
    """Legal moves except turning back."""
    return [h for h in legal_headings(grid, state.pos) if h != state.heading.back]


def is_decision_point(grid: GridMap, state: WorkerState) -> bool:
    """False only in an aisle, where straight ahead is the single non-back option."""
    return forward_options(grid, state) != [state.heading]


def rule_heading(grid: GridMap, state: WorkerState) -> Heading:
    """First legal move in the order straight -> right -> left -> back."""
    h = state.heading
    for candidate in (h, h.right, h.left, h.back):
        if grid.is_free(neighbor(state.pos, candidate)):
            return candidate
    raise ValueError(f"worker at {state.pos} has no free neighbouring cell")


def next_heading(grid: GridMap, state: WorkerState, p: float, rng: np.random.Generator) -> Heading:
    """Heading of the worker's next move.

    Random draws, in order: none in an aisle; at a decision point, one uniform
    draw to choose rule vs. random, then one integer draw if random was chosen.
    """
    if not is_decision_point(grid, state):
        return state.heading
    if rng.random() < p:
        return rule_heading(grid, state)
    options = forward_options(grid, state)
    if not options:  # dead end
        return state.heading.back
    return options[int(rng.integers(len(options)))]


def step_worker(
    grid: GridMap, state: WorkerState, p: float, rng: np.random.Generator
) -> WorkerState:
    """Move one worker by one cell."""
    _check_p(p)
    _check_state(grid, state)
    heading = next_heading(grid, state, p, rng)
    return WorkerState(pos=neighbor(state.pos, heading), heading=heading)


def step_workers(
    grid: GridMap, states: Sequence[WorkerState], p: float, rng: np.random.Generator
) -> list[WorkerState]:
    """Move every worker by one cell, in list order, sharing one generator."""
    return [step_worker(grid, s, p, rng) for s in states]


def _check_p(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"rule adherence p must be in [0, 1], got {p}")


def _check_state(grid: GridMap, state: WorkerState) -> None:
    if not grid.is_free(state.pos):
        raise ValueError(f"worker position {state.pos} is not a free cell")
    if not legal_headings(grid, state.pos):
        raise ValueError(f"worker at {state.pos} is enclosed and cannot move")
