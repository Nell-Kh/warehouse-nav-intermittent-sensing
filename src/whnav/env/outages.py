"""Sensor outage schedules (SPEC §2.7).

A schedule is a boolean array of length ``T`` (the episode deadline). Entry
``i`` is True when the observation the robot uses for its ``(i + 1)``-th
action is missing. During an outage the whole sensor window is unobserved.

Two patterns share one generator:
- scattered: isolated single-step drops, i.e. blocks of length 1;
- consecutive: blocks of exactly ``d`` missing steps.
Blocks never touch: at least one visible step separates any two of them.
Every valid placement of the blocks is equally likely.
"""

from __future__ import annotations

import math

import numpy as np


def block_count(horizon: int, f: float, d: int = 1) -> int:
    """Number of outage blocks: ``f * horizon / d`` rounded half up."""
    _check_inputs(horizon, f, d)
    return math.floor(f * horizon / d + 0.5 + 1e-9)


def is_feasible(horizon: int, f: float, d: int = 1) -> bool:
    """True if the blocks fit in the horizon with a visible step between them.

    The condition is ``n * (d + 1) <= horizon + 1`` for ``n`` blocks. A positive
    ``f`` that rounds to zero blocks is also infeasible, because it would
    silently produce a no-outage episode.
    """
    n = block_count(horizon, f, d)
    if n == 0:
        return f == 0.0
    return n * (d + 1) <= horizon + 1


def outage_schedule(horizon: int, f: float, d: int, rng: np.random.Generator) -> np.ndarray:
    """Schedule with ``block_count(horizon, f, d)`` outages of exactly ``d`` steps.

    Placement is uniform over all valid schedules: choose ``n`` of the
    ``horizon - n*d + 1`` slots that remain once every block is shrunk to a
    point and the mandatory gaps are removed, then expand them again.
    Uses a single ``rng.choice`` call, and none when there are no blocks.
    """
    if not is_feasible(horizon, f, d):
        n = block_count(horizon, f, d)
        raise ValueError(
            f"infeasible outage schedule: horizon={horizon}, f={f}, d={d} gives {n} block(s); "
            f"need at least one block when f > 0 and n*(d+1) <= horizon+1"
        )
    n = block_count(horizon, f, d)
    schedule = np.zeros(horizon, dtype=bool)
    if n == 0:
        return schedule
    slots = np.sort(rng.choice(horizon - n * d + 1, size=n, replace=False))
    for i, slot in enumerate(slots):
        start = int(slot) + i * d
        schedule[start : start + d] = True
    return schedule


def scattered_schedule(horizon: int, f: float, rng: np.random.Generator) -> np.ndarray:
    """Isolated single-step drops: ``round(f * horizon)`` of them, no two adjacent."""
    return outage_schedule(horizon, f, 1, rng)


def consecutive_schedule(horizon: int, f: float, d: int, rng: np.random.Generator) -> np.ndarray:
    """Outages of exactly ``d`` consecutive steps."""
    return outage_schedule(horizon, f, d, rng)


def realized_fraction(schedule: np.ndarray) -> float:
    """Fraction of steps that are actually missing in this schedule."""
    if schedule.ndim != 1 or schedule.size == 0:
        raise ValueError("schedule must be a non-empty 1D array")
    return float(np.count_nonzero(schedule)) / schedule.size


def outage_runs(schedule: np.ndarray) -> list[tuple[int, int]]:
    """``(start, length)`` of every maximal run of missing steps, in order."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, missing in enumerate(schedule):
        if missing and start is None:
            start = i
        elif not missing and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(schedule) - start))
    return runs


def _check_inputs(horizon: int, f: float, d: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if not 0.0 <= f < 1.0:
        raise ValueError(f"missing fraction f must be in [0, 1), got {f}")
    if d < 1:
        raise ValueError(f"outage duration d must be >= 1, got {d}")
