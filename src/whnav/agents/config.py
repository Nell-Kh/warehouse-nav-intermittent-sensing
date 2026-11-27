"""Hyperparameters and JSON configuration for PPO."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PPOConfig:
    """PPO settings for training with complete, equally sized rollouts."""

    seed: int = 1
    total_timesteps: int = 20_000
    learning_rate: float = 2.5e-4
    num_envs: int = 4
    num_steps: int = 125
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    def __post_init__(self) -> None:
        counts = ("total_timesteps", "num_envs", "num_steps", "num_minibatches", "update_epochs")
        for name in counts:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")
        if self.gamma != 0.99:
            raise ValueError("gamma is fixed at 0.99 by the project spec")
        if self.total_timesteps % self.batch_size:
            raise ValueError("total_timesteps must be a multiple of num_envs * num_steps")
        if self.batch_size % self.num_minibatches:
            raise ValueError("num_minibatches must divide the rollout batch size")
        if self.norm_adv and self.minibatch_size < 2:
            raise ValueError("advantage normalization requires at least two samples per minibatch")

    @property
    def batch_size(self) -> int:
        """Number of environment transitions in one rollout."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        """Number of transitions processed in each optimizer step."""
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        """Number of rollouts and PPO updates in the training budget."""
        return self.total_timesteps // self.batch_size


def load_ppo_config(path: str | Path) -> PPOConfig:
    """Load JSON overrides for the PPO defaults from a file."""
    with Path(path).open(encoding="utf-8") as stream:
        settings = json.load(stream)
    if not isinstance(settings, dict):
        raise ValueError("PPO config must be a JSON object")
    return PPOConfig(**settings)
