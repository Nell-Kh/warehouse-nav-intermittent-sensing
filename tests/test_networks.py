"""Tests for the shared encoder and actor/critic heads."""

import numpy as np
import pytest
import torch
from gymnasium import spaces

from whnav.agents.networks import ActorCritic, Encoder


@pytest.mark.parametrize("height,width", [(1, 1), (7, 11), (10, 8), (15, 15)])
def test_encoder_uses_observation_dimensions(height: int, width: int) -> None:
    encoder = Encoder((6, height, width), 3)
    grid = torch.ones(2, 6, height, width)
    scalars = torch.tensor([[1.0, 0.0, 0.5], [0.0, 0.5, 0.25]])

    features = encoder(grid, scalars)

    assert features.shape == (2, 256)
    assert torch.isfinite(features).all()


def test_actor_critic_samples_and_scores_actions() -> None:
    space = spaces.Dict(
        {
            "grid": spaces.Box(0, 1, shape=(6, 7, 11), dtype=np.float32),
            "scalars": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
        }
    )
    model = ActorCritic(space, num_actions=5)
    grid = torch.ones(3, 6, 7, 11)
    scalars = torch.ones(3, 3)
    generator = torch.Generator().manual_seed(7)

    actions, logprobs, entropy, values = model.get_action_and_value(
        grid, scalars, generator=generator
    )
    _, rescored, _, _ = model.get_action_and_value(grid, scalars, actions)

    assert actions.shape == logprobs.shape == entropy.shape == values.shape == (3,)
    assert actions.dtype == torch.long
    assert ((0 <= actions) & (actions < 5)).all()
    torch.testing.assert_close(logprobs, rescored)
    torch.testing.assert_close(values, model.get_value(grid, scalars))
    torch.testing.assert_close(model.actor.weight.norm(dim=1), torch.full((5,), 0.01))
    torch.testing.assert_close(model.critic.weight.norm(), torch.tensor(1.0))
