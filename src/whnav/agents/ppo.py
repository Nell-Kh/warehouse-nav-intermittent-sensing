# Adapted from CleanRL's cleanrl/ppo.py at e421c2e50b81febf639fced51a69e2602593d50d.
# MIT License
#
# Copyright (c) 2019 CleanRL developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Feed-forward PPO on the warehouse observation dictionary."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from whnav.agents.config import PPOConfig, load_ppo_config
from whnav.agents.networks import ActorCritic
from whnav.env.scenario import Scenario, make_scenario
from whnav.env.warehouse_env import WarehouseEnv
from whnav.env.workers import Heading, WorkerState


@dataclass(frozen=True)
class PPOUpdate:
    """Mean minibatch metrics for one update, indexed by environment steps."""

    env_steps: int
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float


@dataclass(frozen=True)
class TrainingResult:
    """Trained policy, update metrics and training episode outcome counts."""

    agent: ActorCritic
    updates: tuple[PPOUpdate, ...]
    episode_outcomes: dict[str, int]
    wall_clock_s: float

    @property
    def env_steps(self) -> int:
        """Number of actual environment transitions collected."""
        return self.updates[-1].env_steps


@dataclass(frozen=True)
class _Batch:
    obs: dict[str, Tensor]
    actions: Tensor
    logprobs: Tensor
    advantages: Tensor
    returns: Tensor
    values: Tensor


def train(
    config: PPOConfig, scenarios: Sequence[Scenario], *, device: str | torch.device = "cpu"
) -> TrainingResult:
    """Train PPO on a pool of same-sized WarehouseEnv scenarios."""
    start = perf_counter()
    device = torch.device(device)
    with closing(
        gym.vector.SyncVectorEnv(
            [partial(WarehouseEnv, scenarios) for _ in range(config.num_envs)],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
    ) as envs:
        with torch.random.fork_rng():
            torch.manual_seed(config.seed)
            agent = ActorCritic(envs.single_observation_space, envs.single_action_space.n)
        agent.to(device)
        updates, outcomes = _train(config, envs, agent, device)
    return TrainingResult(agent, tuple(updates), dict(outcomes), perf_counter() - start)


def _train(
    config: PPOConfig, envs: gym.vector.SyncVectorEnv, agent: ActorCritic, device: torch.device
) -> tuple[list[PPOUpdate], Counter[str]]:
    """Collect rollout batches and update the shared policy and value network."""
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    shuffle_rng = np.random.default_rng(config.seed)
    action_rng = torch.Generator(device=device).manual_seed(config.seed)
    shape = (config.num_steps, config.num_envs)
    obs = {
        key: torch.empty((*shape, *space.shape), dtype=torch.float32, device=device)
        for key, space in envs.single_observation_space.items()
    }
    actions = torch.empty(shape, dtype=torch.long, device=device)
    logprobs = torch.empty(shape, dtype=torch.float32, device=device)
    rewards = torch.empty_like(logprobs)
    values = torch.empty_like(logprobs)
    dones = torch.empty_like(logprobs)
    updates: list[PPOUpdate] = []
    outcomes: Counter[str] = Counter()
    initial_obs, _ = envs.reset(seed=config.seed)
    next_obs = _obs_tensors(initial_obs, device)
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

    for iteration in range(config.num_updates):
        if config.anneal_lr:
            fraction = 1.0 - iteration / config.num_updates
            optimizer.param_groups[0]["lr"] = fraction * config.learning_rate

        for step in range(config.num_steps):
            for key, buffer in obs.items():
                buffer[step] = next_obs[key]
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    **next_obs, generator=action_rng
                )
            actions[step], logprobs[step], values[step] = action, logprob, value

            observation, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            if truncated.any():
                raise RuntimeError("WarehouseEnv episodes must end in a true termination")
            rewards[step] = torch.as_tensor(reward, device=device)
            next_obs = _obs_tensors(observation, device)
            next_done = torch.as_tensor(terminated, dtype=torch.float32, device=device)
            if "final_info" in infos:
                final_info = infos["final_info"]
                outcomes.update(final_info["outcome"][final_info["_outcome"]])

        with torch.no_grad():
            next_value = agent.get_value(**next_obs)
            advantages = _gae(config, rewards, values, dones, next_value, next_done)
        batch = _Batch(
            obs={key: buffer.flatten(0, 1) for key, buffer in obs.items()},
            actions=actions.flatten(),
            logprobs=logprobs.flatten(),
            advantages=advantages.flatten(),
            returns=(advantages + values).flatten(),
            values=values.flatten(),
        )
        env_steps = (iteration + 1) * config.batch_size
        updates.append(_optimize(config, agent, optimizer, batch, shuffle_rng, env_steps))
    return updates, outcomes


def _obs_tensors(
    observation: dict[str, NDArray[np.float32]], device: torch.device
) -> dict[str, Tensor]:
    """Convert the observation batch while preserving its 0–1 scale."""
    return {
        key: torch.as_tensor(array, dtype=torch.float32, device=device)
        for key, array in observation.items()
    }


def _gae(
    config: PPOConfig,
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    next_value: Tensor,
    next_done: Tensor,
) -> Tensor:
    """Compute CleanRL's generalized advantage estimates for one rollout."""
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for step in reversed(range(config.num_steps)):
        if step == config.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            next_values = next_value
        else:
            nextnonterminal = 1.0 - dones[step + 1]
            next_values = values[step + 1]
        discount = config.gamma * nextnonterminal
        delta = rewards[step] + discount * next_values - values[step]
        last_gae = delta + discount * config.gae_lambda * last_gae
        advantages[step] = last_gae
    return advantages


def _optimize(
    config: PPOConfig,
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: _Batch,
    shuffle_rng: np.random.Generator,
    env_steps: int,
) -> PPOUpdate:
    """Apply PPO's clipped policy objective and value loss to one batch."""
    indices = np.arange(config.batch_size)
    metrics: list[Tensor] = []
    for _ in range(config.update_epochs):
        shuffle_rng.shuffle(indices)
        for start in range(0, config.batch_size, config.minibatch_size):
            mb = torch.as_tensor(
                indices[start : start + config.minibatch_size], device=batch.actions.device
            )
            _, logprob, entropy, value = agent.get_action_and_value(
                batch.obs["grid"][mb], batch.obs["scalars"][mb], batch.actions[mb]
            )
            logratio = logprob - batch.logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()

            advantages = batch.advantages[mb]
            if config.norm_adv:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_loss = torch.maximum(
                -advantages * ratio,
                -advantages * ratio.clamp(1 - config.clip_coef, 1 + config.clip_coef),
            ).mean()

            returns = batch.returns[mb]
            value_error = (value - returns).square()
            if config.clip_vloss:
                old_value = batch.values[mb]
                clipped_value = old_value + (value - old_value).clamp(
                    -config.clip_coef, config.clip_coef
                )
                value_error = torch.maximum(value_error, (clipped_value - returns).square())
            value_loss = 0.5 * value_error.mean()
            mean_entropy = entropy.mean()
            loss = policy_loss - config.ent_coef * mean_entropy + config.vf_coef * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
            optimizer.step()
            metrics.append(
                torch.stack(
                    (policy_loss.detach(), value_loss.detach(), mean_entropy.detach(), approx_kl)
                )
            )

        if config.target_kl is not None and approx_kl.item() > config.target_kl:
            break

    policy_loss, value_loss, entropy, approx_kl = torch.stack(metrics).mean(dim=0).tolist()
    return PPOUpdate(env_steps, policy_loss, value_loss, entropy, approx_kl)


def make_demo_scenarios() -> list[Scenario]:
    """Build 20 same-sized smoke-training cases pending the scenario generator."""
    layout = (
        "###########",
        "#.........#",
        "#.##.##.#.#",
        "#.........#",
        "#.##.##.#.#",
        "#.........#",
        "###########",
    )
    conditions = (("none", 0, 0.0),) * 2 + (
        ("scattered", 1, 0.3),
        ("consecutive", 2, 0.3),
        ("consecutive", 4, 0.3),
    ) * 6
    rule_adherence = (0.5, 0.7, 0.9, 1.0)
    return [
        make_scenario(
            layout,
            (1, 1),
            (5, 9),
            [WorkerState((3, 5), Heading.E)],
            p=rule_adherence[index % len(rule_adherence)],
            worker_seed=index,
            pattern=pattern,
            d=duration,
            f=fraction,
            schedule_rng=np.random.default_rng([index, 1]),
            scenario_id=f"ppo-demo-{index:02d}",
            split="train",
            layout_id=0,
            base_id=index,
        )
        for index, (pattern, duration, fraction) in enumerate(conditions)
    ]


def main() -> None:
    """Run the core trainer on the development scenarios and print its metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ppo4.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train(load_ppo_config(args.config), make_demo_scenarios(), device=args.device)
    print(
        json.dumps(
            {
                "env_steps": result.env_steps,
                "episode_outcomes": result.episode_outcomes,
                "wall_clock_s": result.wall_clock_s,
                "last_update": asdict(result.updates[-1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
