"""Gymnasium environment that runs one episode of SPEC §2 (interface 3.1).

Each episode runs one pre-generated :class:`Scenario`. The robot's moves come
from the controller; the workers and the outages come from the scenario only,
so every controller faces exactly the same worker trajectories and sensor
outages (SPEC §2.8).

``info`` returned by ``reset()`` and ``step()``:

=================  ==========  ==========================================================
key                type        meaning
=================  ==========  ==========================================================
scenario_id        str         id of the running scenario
t                  int         steps taken so far (0 after reset)
outcome            str         "running", "success", "collision" or "deadline"
robot              Cell        robot position (also visible in the observation)
outage             bool        True if the returned observation is missing
worker_positions   list[Cell]  true worker positions. Privileged: only privileged A*,
                               auxiliary-loss labels and evaluation code may read it,
                               never a policy input
=================  ==========  ==========================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from whnav.env.grid import Action, Cell, move_robot
from whnav.env.observation import Channel, Scalar, encode_observation
from whnav.env.rules import Outcome, is_collision, step_outcome, step_reward
from whnav.env.scenario import Scenario
from whnav.env.workers import WorkerState, step_workers


class WarehouseEnv(gym.Env):
    """Robot navigation among workers under intermittent sensing.

    Observation: ``{"grid": (6, H, W), "scalars": (3,)}``, both float32 in [0, 1]
    (SPEC §2.6). Actions: ``Discrete(5)`` = N, S, E, W, WAIT. ``terminated`` is
    True on success, collision and deadline; ``truncated`` is always False,
    because the deadline is part of the task: time left is in the observation
    (SPEC §2.4).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, scenarios: Sequence[Scenario], render_mode: str | None = None) -> None:
        if not scenarios:
            raise ValueError("scenarios must not be empty")
        shape = (scenarios[0].grid.height, scenarios[0].grid.width)
        for sc in scenarios:
            if (sc.grid.height, sc.grid.width) != shape:
                raise ValueError(
                    f"all scenarios must share one grid shape {shape}; "
                    f"{sc.scenario_id} has {(sc.grid.height, sc.grid.width)}"
                )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.scenarios: tuple[Scenario, ...] = tuple(scenarios)
        self.render_mode = render_mode
        height, width = shape
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0.0, 1.0, (len(Channel), height, width), np.float32),
                "scalars": spaces.Box(0.0, 1.0, (len(Scalar),), np.float32),
            }
        )
        self.action_space = spaces.Discrete(len(Action))

        self._sc: Scenario | None = None
        self._robot: Cell = (0, 0)
        self._workers: list[WorkerState] = []
        self._worker_rng: np.random.Generator | None = None
        self._schedule = np.zeros(0, dtype=bool)
        self._t = 0
        self._obs_index = 0
        self._outcome = Outcome.RUNNING

    # --- Gymnasium API ---

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Start an episode.

        ``options={"scenario": s}`` runs ``s``. Otherwise one of ``scenarios`` is
        drawn with ``self.np_random``, so the order of training episodes follows
        the seed. ``self.np_random`` is used for nothing else.
        """
        super().reset(seed=seed)
        if options and "scenario" in options:
            sc = options["scenario"]
            if (sc.grid.height, sc.grid.width) != self.observation_space["grid"].shape[1:]:
                raise ValueError("scenario grid shape does not match this environment")
        else:
            sc = self.scenarios[int(self.np_random.integers(len(self.scenarios)))]

        self._sc = sc
        self._robot = sc.start
        self._workers = list(sc.workers)
        # A fresh generator per episode keeps worker trajectories identical for
        # every controller and independent of how many episodes ran before.
        self._worker_rng = np.random.default_rng(sc.worker_seed)
        self._schedule = np.array(sc.schedule, dtype=bool)
        self._t = 0
        self._obs_index = 0
        self._outcome = Outcome.RUNNING
        return self._observe(), self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one step (SPEC §2.2): all moves use start-of-step positions, then collisions."""
        sc = self._require_scenario()
        if self._outcome.is_terminal:
            raise RuntimeError("episode is over; call reset()")
        act = Action(int(action))

        self._t += 1
        robot_before = self._robot
        workers_before = [w.pos for w in self._workers]
        self._robot = move_robot(sc.grid, robot_before, act)
        self._workers = step_workers(sc.grid, self._workers, sc.p, self._worker_rng)
        workers_after = [w.pos for w in self._workers]

        collided = is_collision(robot_before, self._robot, workers_before, workers_after)
        self._outcome = step_outcome(self._robot, sc.goal, collided, self._t, sc.horizon)
        reward = step_reward(self._outcome, sc.horizon)

        # Observation after step t uses schedule entry t (it serves action t + 1).
        # On the terminal step nobody acts on it, so the last entry is reused.
        self._obs_index = min(self._t, sc.horizon - 1)
        obs = self._observe()
        return obs, reward, self._outcome.is_terminal, False, self._info()

    def render(self) -> str | None:
        """Text map of the current state, for ``render_mode="ansi"``.

        ``#`` shelf or wall, ``.`` free, ``W`` worker, ``R`` robot, ``G`` goal,
        ``*`` robot on a worker's cell (a collision).
        """
        if self.render_mode != "ansi":
            return None
        sc = self._require_scenario()
        rows = [list(r) for r in sc.layout]
        gr, gc = sc.goal
        rows[gr][gc] = "G"
        for w in self._workers:
            rows[w.pos[0]][w.pos[1]] = "W"
        rr, rc = self._robot
        rows[rr][rc] = "*" if rows[rr][rc] == "W" else "R"
        status = (
            f"t={self._t}/{sc.horizon} outcome={self._outcome.value} "
            f"sensor={'OFF' if sc.schedule[self._obs_index] else 'on'}"
        )
        return "\n".join("".join(r) for r in rows) + "\n" + status

    # --- helpers ---

    def _require_scenario(self) -> Scenario:
        if self._sc is None:
            raise RuntimeError("call reset() before step() or render()")
        return self._sc

    def _observe(self) -> dict[str, np.ndarray]:
        sc = self._require_scenario()
        return encode_observation(
            sc.grid,
            self._robot,
            sc.goal,
            [w.pos for w in self._workers],
            self._schedule,
            self._obs_index,
        )

    def _info(self) -> dict[str, Any]:
        sc = self._require_scenario()
        return {
            "scenario_id": sc.scenario_id,
            "t": self._t,
            "outcome": self._outcome.value,
            "robot": self._robot,
            "outage": bool(sc.schedule[self._obs_index]),
            "worker_positions": [w.pos for w in self._workers],
        }
