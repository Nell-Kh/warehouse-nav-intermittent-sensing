# Plan

Working checklist for the timeline in [SPEC.md §8](SPEC.md#8-timeline).
There are no fixed roles. Each task has exactly one owner from the moment someone starts it:
1. Open a GitHub issue for the task (or find the existing one) and assign yourself before writing code.
2. If an issue is already assigned, pick another task or talk to the owner first.
3. When the PR is merged, tick the box and write your name after the task, in that same PR.

## Week-1 decisions (all three agree, then record in SPEC)

| Decision | Proposed | Agreed |
|---|---|---|
| Robot moves into a shelf or wall | Stays in place | ☐ |
| Worker speed (cells per step) | 1 | ☐ |
| Observation tensor layout | Full-map channels vs. robot-centered crop: decide | ☐ |
| Outage schedule generation | Exact `round(f·T)` drops; see SPEC §2.7 | ☐ |
| Presentation length | Ask the lecturer (guideline says 15 min) | ☐ |

## Week 1 · Nov 8–14

- [x] Repo, spec, plan, contribution rules (Nell)
- [ ] Week-1 decisions above (all)
- [ ] Grid, robot, workers, collisions, deadline, reward
- [ ] Observation encoding per SPEC §2.6
- [ ] Sensor outages, scattered and consecutive, same `f`
- [ ] Scenario generator with seeds; four splits with held-out layouts and seeds
- [ ] Tests: collision (same cell, swap), visibility, outage, deadline, terminal handling
- [ ] Test: a scenario replays identically from its seed
- [ ] Last-seen A* and privileged A*

**Milestone:** a scenario replays identically across controllers; all tests pass; both A* references run.

## Week 2 · Nov 15–21

- [ ] Related-work list, 5–8 papers with one line each (all)
- [ ] Freeze heuristic
- [ ] Motion-predicting planner: tracker, `p̂` fitted on training mix, time-expanded A* with wait
- [ ] Calibration pilot on development scenarios, SPEC §9
- [ ] PPO 4-frame solves a small no-outage task
- [ ] Measure training time per run, then set the tuning budget
- [ ] Record pilot results in SPEC §10 with date and reason

**Gate:** if anything slips, shrink the warehouse and the number of workers first.

## Week 3 · Nov 22–28

- [ ] Validation learning-curve logging, shared by all learned methods
- [ ] Recurrent PPO (LSTM), CleanRL input handling and network adapted
- [ ] First results with outages
- [ ] Report: environment section

## Week 4 · Nov 29–Dec 5

- [ ] Method 3: auxiliary occupancy head, masked weighted BCE, `λ = 0` reproduces method 2
- [ ] Test: same seed gives identical starting weights and scenario order for methods 2 and 3

## Week 5 · Dec 6–12

- [ ] Optuna tuning, same trials and steps for every method
- [ ] 15 main runs (methods 1–3 × 5 seeds), wall-clock logged

## Week 6 · Dec 13–19

- [ ] Test evaluation: `d × p` heatmap, scattered comparison, no-outage control
- [ ] Prediction quality: AP and AUROC on freeze-heuristic replays and on own trajectories
- [ ] Data-efficiency plots: learning curves, steps-to-threshold, area under the curve (SPEC §6.1)
- [ ] Statistics per SPEC §6.3, individual seeds shown next to CIs

## Week 7 · Dec 20–26

- [ ] Report, 8–20 pages, Arial 12, 1.5 spacing, 1-inch margins (all)
- [ ] Presentation (all)
- [ ] Demo video: robot during an outage, with vs. without memory
- [ ] Final check: a fresh clone reproduces one result from the README alone (all)

## What the guideline grades, and where we cover it

| Guideline item | Where |
|---|---|
| Problem formulation: states, actions, rewards, dynamics | SPEC §2 and the formal POMDP in §2.9, report "Problem Formulation" |
| Algorithm selection, justified | SPEC §3 "Why these algorithms", report "Methodology" |
| Thorough hyperparameter search | Optuna, SPEC §5 |
| Evaluation metrics and comparison of approaches | SPEC §6, including data efficiency (§6.1) |
| Comparison to existing work, limitations | Related-work list (week 2), report "Discussion" |
| Clean, organized, commented code | CONTRIBUTING.md rules, CI lint |
| Reproducibility | Seeds, pre-generated scenarios, README instructions |
| Version control | Feature branches and reviewed PRs, CONTRIBUTING.md |
| All members contribute | Each person commits their own work through PRs |
