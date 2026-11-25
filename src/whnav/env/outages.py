"""Sensor outage schedules (SPEC §2.7).

A schedule is a boolean array of length ``T`` (the episode deadline). Entry
``i`` is True when the observation the robot uses for its ``(i + 1)``-th
action is missing. During an outage the whole sensor window is unobserved.

Two patterns share one generator:
- scattered: isolated single-step drops, i.e. blocks of length 1;
- consecutive: blocks of exactly ``d`` missing steps.
Blocks never touch: at least one visible step separates any two of them.
Every valid placement of the blocks is equally likely.

The number of blocks is ``x = f * T / d`` rounded at random: up with
probability equal to its fractional part, down otherwise. So the expected
missing fraction is exactly ``f`` for every ``d``, and the comparison across
``d`` at one ``f`` is not biased by rounding. Each schedule still differs from
``f`` by less than ``d`` steps.
"""

from __future__ import annotations

import math

import numpy as np

_EPS = 1e-9  # absorbs float error such as 0.1 * 30 = 3.0000000000000004


def expected_blocks(horizon: int, f: float, d: int = 1) -> float:
    """Expected number of outage blocks, ``f * horizon / d``."""
    _check_inputs(horizon, f, d)
    return f * horizon / d


def block_count_range(horizon: int, f: float, d: int = 1) -> tuple[int, int]:
    """Smallest and largest number of blocks a schedule can have."""
    x = expected_blocks(horizon, f, d)
    low = math.floor(x + _EPS)
    high = low if x - low < _EPS else low + 1
    return low, high


def is_feasible(horizon: int, f: float, d: int = 1) -> bool:
    """True if every possible schedule fits in the horizon.

    Needs at least one block whenever ``f > 0`` (``f * T / d >= 1``), so an
    outage condition never silently becomes a no-outage episode, and room for
    the largest block count with a visible step between blocks:
    ``high * (d + 1) <= horizon + 1``.
    """
    low, high = block_count_range(horizon, f, d)
    if f == 0.0:
        return True
    return low >= 1 and high * (d + 1) <= horizon + 1


def draw_block_count(horizon: int, f: float, d: int, rng: np.random.Generator) -> int:
    """Number of blocks for one schedule: ``f * T / d`` rounded at random.

    Uses one ``rng.random()`` call when the expected count is not a whole
    number, and none otherwise.
    """
    low, high = block_count_range(horizon, f, d)
    if low == high:
        return low
    frac = expected_blocks(horizon, f, d) - low
    return high if rng.random() < frac else low


def outage_schedule(horizon: int, f: float, d: int, rng: np.random.Generator) -> np.ndarray:
    """Schedule with ``draw_block_count`` outages of exactly ``d`` steps.

    Placement is uniform over all valid schedules with that many blocks:
    choose ``n`` of the ``horizon - n*d + 1`` slots that remain once every
    block is shrunk to a point and the mandatory gaps are removed, then expand
    them again. Random draws, in order: the block-count draw (if any), then
    one ``rng.choice`` call (none when there are no blocks).
    """
    if not is_feasible(horizon, f, d):
        low, high = block_count_range(horizon, f, d)
        raise ValueError(
            f"infeasible outage schedule: horizon={horizon}, f={f}, d={d} gives {low}-{high} "
            f"block(s); need f*T/d >= 1 when f > 0, and blocks*(d+1) <= horizon+1"
        )
    n = draw_block_count(horizon, f, d, rng)
    schedule = np.zeros(horizon, dtype=bool)
    if n == 0:
        return schedule
    slots = np.sort(rng.choice(horizon - n * d + 1, size=n, replace=False))
    for i, slot in enumerate(slots):
        start = int(slot) + i * d
        schedule[start : start + d] = True
    return schedule


def scattered_schedule(horizon: int, f: float, rng: np.random.Generator) -> np.ndarray:
    """Isolated single-step drops, ``f * horizon`` of them on average, no two adjacent."""
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
