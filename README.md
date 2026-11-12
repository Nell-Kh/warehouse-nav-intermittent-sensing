# Warehouse Navigation under Intermittent Sensing

Reinforcement Learning course project, University of Haifa.

**Research question:** under intermittent sensing, when does predictive memory improve warehouse navigation beyond short observation histories and explicit motion-based planning?

A robot navigates a grid warehouse among workers who follow a turn rule with probability `p` (rule adherence). Its 7×7 sensor drops out, in single steps or in outages of duration `d`. We compare PPO with a 4-frame history, recurrent PPO, recurrent PPO with an auxiliary occupancy-prediction loss, and a motion-predicting planner, across `d × p`.

Full design: [docs/SPEC.md](docs/SPEC.md) · Progress: [docs/PLAN.md](docs/PLAN.md) · Team rules: [CONTRIBUTING.md](CONTRIBUTING.md)

## Status

Week 1: environment and scenario generator in progress.

## Setup

Requires Python 3.10 or newer.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the tests

```
pytest -q
```

## Repository layout

```
src/whnav/
  env/        grid, workers, sensor outages, scenarios
  planners/   last-seen A*, privileged A*, freeze heuristic, motion-predicting planner
  agents/     PPO 4-frame, recurrent PPO, recurrent PPO + auxiliary loss
  eval/       metrics, statistics, plots
configs/      experiment and hyperparameter settings
tests/        unit and replay tests
docs/         spec and plan
```

## Team

| Name | GitHub |
|---|---|
| Nell Khoury | [@Nell-Kh](https://github.com/Nell-Kh) |

Teammates add their own row in their first pull request.
