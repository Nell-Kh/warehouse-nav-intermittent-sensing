"""Collision, termination and reward rules (SPEC §2.4, §2.5).

Step ``t`` counts the steps taken so far, including the current one, so the
first step is ``t = 1`` and the last possible step is ``t = T``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from whnav.env.grid import Cell

SUCCESS_REWARD = 1.0
COLLISION_REWARD = -1.0
TOTAL_STEP_COST = 0.5  # running out the whole deadline costs exactly this much


class Outcome(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    COLLISION = "collision"
    DEADLINE = "deadline"

    @property
    def is_terminal(self) -> bool:
        return self is not Outcome.RUNNING


def is_collision(
    robot_before: Cell,
    robot_after: Cell,
    workers_before: Sequence[Cell],
    workers_after: Sequence[Cell],
) -> bool:
    """True if the robot collides with any worker during this step.

    Two rules, complete for worker speed 1 (SPEC §2.4):
    1. the robot and a worker end the step in the same cell;
    2. the robot and a worker swap cells.
    Workers are anonymous, but ``workers_before[i]`` and ``workers_after[i]``
    must be the same worker.
    """
    if len(workers_before) != len(workers_after):
        raise ValueError("workers_before and workers_after must have the same length")
    for before, after in zip(workers_before, workers_after, strict=True):
        if after == robot_after:
            return True
        if before == robot_after and after == robot_before and robot_before != robot_after:
            return True
    return False


def step_outcome(robot_after: Cell, goal: Cell, collided: bool, t: int, horizon: int) -> Outcome:
    """Outcome of step ``t`` (1-based) in an episode with deadline ``horizon`` (= T).

    Order matters: a collision on the step the goal is reached counts as a
    collision, and reaching the goal on step T counts as success (SPEC §2.4).
    """
    if not 1 <= t <= horizon:
        raise ValueError(f"step t={t} outside 1..{horizon}")
    if collided:
        return Outcome.COLLISION
    if robot_after == goal:
        return Outcome.SUCCESS
    if t == horizon:
        return Outcome.DEADLINE
    return Outcome.RUNNING


def step_reward(outcome: Outcome, horizon: int) -> float:
    """Reward for one step: ``-0.5/T`` every step, plus ``+1`` or ``-1`` on success or collision."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    reward = -TOTAL_STEP_COST / horizon
    if outcome is Outcome.SUCCESS:
        reward += SUCCESS_REWARD
    elif outcome is Outcome.COLLISION:
        reward += COLLISION_REWARD
    return reward
