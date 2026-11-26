"""Estimate the auxiliary BCE class weight from seeded random warehouse episodes."""

import argparse
import json
from collections.abc import Sequence
from contextlib import closing

import numpy as np

from whnav.agents.aux_target import AuxTargetWrapper
from whnav.env.scenario import Scenario, make_scenario
from whnav.env.warehouse_env import WarehouseEnv
from whnav.env.workers import Heading, WorkerState


def make_demo_scenarios() -> list[Scenario]:
    """Build the same temporary 20-scenario pool as the PPO demo."""
    # Keep this copy until the training split lands so RL-6 can run before RL-1 merges.
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


def count_masked_cells(scenarios: Sequence[Scenario], episodes: int, seed: int) -> tuple[int, int]:
    """Count positives and negatives in the observations used to choose actions."""
    positives = masked_cells = 0
    with closing(AuxTargetWrapper(WarehouseEnv(scenarios))) as env:
        env.action_space.seed(seed)
        for episode in range(episodes):
            scenario = scenarios[episode % len(scenarios)]
            _, info = env.reset(options={"scenario": scenario})
            terminated = truncated = False
            while not (terminated or truncated):
                mask = info["aux_mask"]
                positives += int(np.count_nonzero(info["aux_labels"][mask]))
                masked_cells += int(np.count_nonzero(mask))
                # Use the freeze policy once it is available.
                _, _, terminated, truncated, info = env.step(int(env.action_space.sample()))
    return positives, masked_cells - positives


def main() -> None:
    """Print the sampled counts and negative-to-positive ratio as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1_000, help="episodes to sample")
    parser.add_argument("--seed", type=int, default=0, help="random action seed")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    scenarios = make_demo_scenarios()
    positives, negatives = count_masked_cells(scenarios, args.episodes, args.seed)
    if not positives:
        parser.error("no positive masked cells sampled. Try more episodes or a different seed.")
    print(
        json.dumps(
            {
                "episodes": args.episodes,
                "scenario_pool_size": len(scenarios),
                "seed": args.seed,
                "positives": positives,
                "negatives": negatives,
                "pos_weight": negatives / positives,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
