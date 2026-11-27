"""Tests for PPO configuration and rollout sizing."""

from pathlib import Path

import pytest

from whnav.agents.config import PPOConfig, load_ppo_config


def test_load_project_config() -> None:
    config = load_ppo_config(Path(__file__).parents[1] / "configs" / "ppo4.json")

    assert config.gamma == 0.99
    assert config.num_updates * config.batch_size == config.total_timesteps == 20_000
    assert config.minibatch_size * config.num_minibatches == config.batch_size


def test_load_json_overrides(tmp_path: Path) -> None:
    path = tmp_path / "ppo.json"
    path.write_text('{"seed": 7, "target_kl": 0.03}', encoding="utf-8")

    assert load_ppo_config(path) == PPOConfig(seed=7, target_kl=0.03)


def test_reject_non_object_config(tmp_path: Path) -> None:
    path = tmp_path / "ppo.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        load_ppo_config(path)


def test_reject_incomplete_rollout() -> None:
    with pytest.raises(ValueError, match="total_timesteps"):
        PPOConfig(total_timesteps=20_001)


def test_reject_uneven_minibatches() -> None:
    with pytest.raises(ValueError, match="num_minibatches"):
        PPOConfig(num_minibatches=3)


def test_reject_non_spec_discount() -> None:
    with pytest.raises(ValueError, match="gamma is fixed"):
        PPOConfig(gamma=0.95)
