"""Shared schemas for episode results and validation learning curves."""

from dataclasses import dataclass

from whnav.eval.controller import METHODS as METHODS


@dataclass(frozen=True)
class EpisodeResult:
    """One completed evaluation episode.

    ``train_seed`` is -1 for deterministic controllers, which have no
    checkpoint. ``episode_return`` is exported as the CSV column ``return``.
    """

    method: str
    train_seed: int
    checkpoint: str | None
    scenario_id: str
    split: str
    base_id: int
    layout_id: int
    pattern: str
    d: int
    p: float
    f: float
    f_realized: float
    outcome: str
    steps: int
    episode_return: float
    horizon: int
    shortest_path: int
    wall_clock_s: float


@dataclass(frozen=True)
class LearningCurveRow:
    """Validation metrics at a single training step for one method and seed."""

    method: str
    train_seed: int
    env_steps: int
    success_rate: float
    collision_rate: float
    deadline_rate: float
    mean_return: float
    n_episodes: int
    wall_clock_s: float
