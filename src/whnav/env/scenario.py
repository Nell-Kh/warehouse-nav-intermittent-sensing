"""Pre-generated scenarios (SPEC §2.8, interface 3.1 of the work packages).

A scenario fixes everything about one episode except the controller's actions:
the layout, start and goal, the workers' initial states, the rule adherence
``p``, the seed of the workers' own random generator, and the outage schedule.
Running the same scenario twice gives identical worker trajectories and
observations for every controller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from whnav.env.grid import Cell, GridMap, deadline, shortest_path_length
from whnav.env.outages import outage_runs, outage_schedule, realized_fraction
from whnav.env.workers import Heading, WorkerState, legal_headings

SPLITS = ("train", "dev", "val", "test")
PATTERNS = ("none", "scattered", "consecutive")


@dataclass(frozen=True)
class Scenario:
    """One fully specified episode. Immutable and hashable.

    ``schedule`` has length ``horizon``; entry ``i`` is True when the
    observation used for action ``i + 1`` is missing (SPEC §2.7).
    """

    scenario_id: str
    split: str
    layout_id: int
    base_id: int
    layout: tuple[str, ...]
    start: Cell
    goal: Cell
    workers: tuple[WorkerState, ...]
    p: float
    worker_seed: int
    pattern: str
    d: int
    f: float
    schedule: tuple[bool, ...]
    shortest_path: int
    horizon: int

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if self.pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}, got {self.pattern!r}")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        grid = self.grid
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if not grid.is_free(cell):
                raise ValueError(f"{name} {cell} is not a free cell")
        if self.start == self.goal:
            raise ValueError("start and goal must differ")
        length = shortest_path_length(grid, self.start, self.goal)
        if length is None:
            raise ValueError("goal is not reachable from start")
        if self.shortest_path != length:
            raise ValueError(f"shortest_path is {self.shortest_path}, BFS gives {length}")
        if self.horizon != deadline(length):
            raise ValueError(f"horizon must be 2 * shortest_path = {deadline(length)}")
        if len(self.schedule) != self.horizon:
            raise ValueError(f"schedule has length {len(self.schedule)}, expected {self.horizon}")
        for w in self.workers:
            if not grid.is_free(w.pos) or not legal_headings(grid, w.pos):
                raise ValueError(f"worker at {w.pos} is not on a free, connected cell")
        self._check_pattern()

    def _check_pattern(self) -> None:
        runs = [length for _, length in outage_runs(np.array(self.schedule, dtype=bool))]
        if self.pattern == "none":
            if self.d != 0 or self.f != 0.0 or runs:
                raise ValueError("pattern 'none' needs d = 0, f = 0 and no outages")
            return
        expected_d = 1 if self.pattern == "scattered" else self.d
        if self.d != expected_d or self.d < 1:
            raise ValueError(f"pattern {self.pattern!r} needs d = {expected_d}, got {self.d}")
        if not 0.0 < self.f < 1.0:
            raise ValueError(f"pattern {self.pattern!r} needs 0 < f < 1, got {self.f}")
        if not runs or any(length != self.d for length in runs):
            raise ValueError(f"every outage must last exactly d = {self.d} steps, got {runs}")

    @cached_property
    def grid(self) -> GridMap:
        """The static map, built once from ``layout``."""
        return GridMap.from_strings(list(self.layout))

    @cached_property
    def f_realized(self) -> float:
        """Fraction of steps whose observation is actually missing."""
        return realized_fraction(np.array(self.schedule, dtype=bool))

    def to_json(self) -> str:
        """One-line JSON, for one-scenario-per-line JSONL files."""
        data = asdict(self)
        data["workers"] = [[list(w["pos"]), int(w["heading"])] for w in data["workers"]]
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Scenario:
        data = json.loads(line)
        data["layout"] = tuple(data["layout"])
        data["start"] = tuple(data["start"])
        data["goal"] = tuple(data["goal"])
        data["workers"] = tuple(
            WorkerState(pos=tuple(pos), heading=Heading(h)) for pos, h in data["workers"]
        )
        data["schedule"] = tuple(bool(x) for x in data["schedule"])
        return cls(**data)


def make_scenario(
    layout: list[str] | tuple[str, ...],
    start: Cell,
    goal: Cell,
    workers: list[WorkerState] | tuple[WorkerState, ...] = (),
    *,
    p: float = 1.0,
    worker_seed: int = 0,
    pattern: str = "none",
    d: int = 0,
    f: float = 0.0,
    schedule_rng: np.random.Generator | None = None,
    schedule: list[bool] | tuple[bool, ...] | None = None,
    scenario_id: str = "manual",
    split: str = "dev",
    layout_id: int = -1,
    base_id: int = -1,
) -> Scenario:
    """Build a scenario, computing ``shortest_path``, ``horizon`` and the schedule.

    Give either ``schedule`` (an explicit schedule, for tests) or
    ``schedule_rng`` (draws one with :func:`outage_schedule`). Pattern
    ``"none"`` needs neither. For ``"scattered"``, ``d`` is set to 1.
    """
    grid = GridMap.from_strings(list(layout))
    length = shortest_path_length(grid, start, goal)
    if length is None:
        raise ValueError("goal is not reachable from start")
    horizon = deadline(length)
    if pattern == "scattered":
        d = 1
    if schedule is None:
        if pattern == "none":
            schedule = [False] * horizon
        elif schedule_rng is None:
            raise ValueError("give schedule or schedule_rng for an outage pattern")
        else:
            schedule = outage_schedule(horizon, f, d, schedule_rng).tolist()
    return Scenario(
        scenario_id=scenario_id,
        split=split,
        layout_id=layout_id,
        base_id=base_id,
        layout=tuple(layout),
        start=tuple(start),
        goal=tuple(goal),
        workers=tuple(workers),
        p=p,
        worker_seed=worker_seed,
        pattern=pattern,
        d=d,
        f=f,
        schedule=tuple(bool(x) for x in schedule),
        shortest_path=length,
        horizon=horizon,
    )
