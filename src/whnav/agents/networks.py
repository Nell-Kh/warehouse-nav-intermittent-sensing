# Orthogonal initialization follows CleanRL's MIT-licensed ppo.py,
# commit e421c2e50b81febf639fced51a69e2602593d50d. The upstream LICENSE carries the notice.
"""Shared convolutional encoder and feed-forward PPO policy."""

import math

import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Categorical


def layer_init(layer: nn.Conv2d | nn.Linear, std: float = math.sqrt(2)) -> nn.Module:
    """Initialize convolutional and linear layers with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(nn.Module):
    """Encode a batch of full-map grids and scalar vectors."""

    def __init__(
        self, grid_shape: tuple[int, int, int], n_scalars: int, hidden: int = 256
    ) -> None:
        super().__init__()
        channels, height, width = grid_shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, channels, height, width)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(flat_size + n_scalars, hidden)), nn.ReLU())

    def forward(self, grid: Tensor, scalars: Tensor) -> Tensor:
        """Return one feature vector per observation."""
        return self.fc(torch.cat((self.conv(grid), scalars), dim=1))


class ActorCritic(nn.Module):
    """Categorical policy and scalar value function sharing one encoder."""

    def __init__(self, observation_space: spaces.Dict, num_actions: int) -> None:
        super().__init__()
        grid_shape = observation_space["grid"].shape
        (n_scalars,) = observation_space["scalars"].shape
        hidden = 256
        self.encoder = Encoder(grid_shape, n_scalars, hidden)
        self.actor = layer_init(nn.Linear(hidden, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def forward(self, grid: Tensor, scalars: Tensor) -> tuple[Tensor, Tensor]:
        """Return action logits and state values, with one row per observation."""
        features = self.encoder(grid, scalars)
        return self.actor(features), self.critic(features).squeeze(-1)

    def get_value(self, grid: Tensor, scalars: Tensor) -> Tensor:
        """Estimate the value of each observation."""
        return self.critic(self.encoder(grid, scalars)).squeeze(-1)

    def get_action_and_value(
        self,
        grid: Tensor,
        scalars: Tensor,
        action: Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Score given actions or sample actions using the supplied generator."""
        logits, value = self(grid, scalars)
        policy = Categorical(logits=logits)
        if action is None:
            if generator is None:
                raise ValueError("sampling actions requires a torch.Generator")
            action = torch.multinomial(policy.probs, 1, generator=generator).squeeze(-1)
        return action, policy.log_prob(action), policy.entropy(), value
