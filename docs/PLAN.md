# Plan

Decisions, finished work and how the course guideline is covered.

## Where the work is tracked

From 2026-11-25 every task is a GitHub issue (`ENV-*`, `RL-*`, `EVAL-*`, `TEAM-*`) with an owner, a due date and what "done" means. The timeline is in [SPEC §8](SPEC.md#8-timeline) and the lanes in [SPEC §7](SPEC.md#7-team).

## Decisions (2026-11-25)

| Decision | Choice | SPEC |
|---|---|---|
| Robot moves into a shelf or wall | Stays in place | §2.2 |
| Worker speed | 1 cell per step | §2.3 |
| Observation layout | Full-map channels | §2.6 |
| Outage schedule | Uniform placement, block count rounded at random | §2.7 |
| Worker start positions | Not within Manhattan distance 2 of the robot's start | §2.8 |
| No-outage share of training episodes | 10% | §2.8 |
| Previous action as network input | No | §5 |
| Evaluation action of learned methods | Sampled, seeded per scenario | §5 |
| Presentation length | 15 minutes | §8 |
| PR approval | Any teammate in their own lane; Nell for environment, interfaces and SPEC | CONTRIBUTING |

## Done before the lanes

- [x] Repo, spec, plan, contribution rules (Nell)
- [x] Grid, robot, workers, collisions, deadline, reward (Nell)
- [x] Sensor outages, scattered and consecutive, same `f` (Nell)
- [x] Observation encoding per SPEC §2.6 (Nell)

## What the guideline grades, and where we cover it

| Guideline item | Where |
|---|---|
| Problem formulation: states, actions, rewards, dynamics | SPEC §2 and the formal POMDP in §2.9, report "Problem Formulation" |
| Algorithm selection, justified | SPEC §3 "Why these algorithms", report "Methodology" |
| Thorough hyperparameter search | Optuna, SPEC §5 |
| Evaluation metrics and comparison of approaches | SPEC §6, including data efficiency (§6.1) |
| Comparison to existing work, limitations | Related-work list (TEAM-2), report "Discussion" |
| Clean, organized, commented code | CONTRIBUTING.md rules, CI lint |
| Reproducibility | Seeds, pre-generated scenarios, README instructions |
| Version control | Feature branches and reviewed PRs, CONTRIBUTING.md |
| All members contribute | Each person commits their own work through PRs |
