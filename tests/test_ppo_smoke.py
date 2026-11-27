"""Rollout, optimizer and CPU reproducibility tests against WarehouseEnv."""

from collections.abc import Iterator
from math import isfinite

import pytest
import torch

from whnav.agents.config import PPOConfig
from whnav.agents.ppo import TrainingResult, make_demo_scenarios, train


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    """Keep small convolutions from oversubscribing test runners."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def assert_training_completed(result: TrainingResult, config: PPOConfig) -> None:
    """Check the step budget, losses and same-step terminal info."""
    assert result.env_steps == config.total_timesteps
    assert len(result.updates) == config.num_updates
    assert result.episode_outcomes
    assert result.episode_outcomes.keys() <= {"success", "collision", "deadline"}
    for update in result.updates:
        metrics = (update.policy_loss, update.value_loss, update.entropy, update.approx_kl)
        assert all(isfinite(metric) for metric in metrics)


def test_short_ppo_training() -> None:
    config = PPOConfig(total_timesteps=192, num_envs=4, num_steps=16, update_epochs=1)

    result = train(config, make_demo_scenarios())

    assert_training_completed(result, config)


def test_ppo_trains_for_twenty_thousand_steps() -> None:
    config = PPOConfig(num_envs=8, num_minibatches=8, update_epochs=1)
    scenarios = make_demo_scenarios()

    assert len(scenarios) == 20
    assert sum(scenario.pattern == "none" for scenario in scenarios) == 2
    assert_training_completed(train(config, scenarios), config)


def test_same_seed_reproduces_cpu_losses() -> None:
    config = PPOConfig(total_timesteps=64, num_envs=2, num_steps=16, update_epochs=2)
    scenarios = make_demo_scenarios()
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        first = train(config, scenarios, device="cpu")
        second = train(config, scenarios, device="cpu")
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    assert len(first.updates) == 2
    assert first.updates == second.updates
    assert first.episode_outcomes == second.episode_outcomes
