# SPEC — Warehouse Navigation under Intermittent Sensing

RL course project (Prof. Teddy Lazebnik, University of Haifa). Group of 3.

**Status:** research scope and methods frozen on 2026-11-05. The only open items are the week-1 decisions (§2) and the parameters in §10. The pilot sets the parameters, and they're recorded there with the date and the reason. Don't add methods. Every change to this file is logged in §12.

---

## 1. Research question

> Under intermittent sensing, when does predictive memory improve warehouse navigation beyond short observation histories and explicit motion-based planning?

- **Main contribution:** the controlled experiment, i.e. how outage duration × rule adherence affects each method.
- **Proposed variant:** recurrent PPO with an auxiliary occupancy-prediction loss. It's a variant, not a new algorithm.
- **Results either way:** if the auxiliary method doesn't win, or the freeze heuristic does, that is still a valid finding.

---

## 2. Environment

### 2.1 Grid and layouts
- Fixed grid size (§10). Shelf arrangements vary between scenarios.
- Each scenario uses a reachable start–goal pair with a minimum distance (§10).

### 2.2 Robot
- Actions: `N`, `S`, `E`, `W`, `wait`. The robot moves 1 cell per step.
- The robot and the workers move simultaneously: every new position is computed from the positions at the start of the step, then the collision checks in §2.4 run.
- *Week-1 decision:* what happens when the robot moves into a shelf or wall (proposed: it stays in place).

### 2.3 Workers
- **Speed:** constant between decision points (§10). In aisles, workers always move forward.
- *Week-1 decision:* worker speed in cells per step. Proposed: 1 cell/step. With speed > 1 a worker can cross the robot's path inside one step, which the two rules in §2.4 don't catch; choosing speed > 1 therefore also requires defining a sub-step collision check before implementation.
- **Decision points** (intersections, corners, dead ends):
  - with probability **p**, follow the rule **straight → right → left → back**, taking the first legal move;
  - otherwise, pick uniformly among legal moves, excluding back unless it's a dead end.
- **p** is called *rule adherence*. It isn't a direct measure of predictability.
- Workers don't react to the robot.
- Workers can share a cell (no worker–worker collisions). A cell counts as occupied if at least one worker is in it.
- Worker count is fixed (§10).

### 2.4 Collisions and termination
- **Collision:**
  - the robot and a worker end the step in the same cell, or
  - the robot and a worker swap cells within one step.
  - (These two rules are complete only for worker speed 1, see §2.3.)
- **Outcomes:** `success` / `collision` / `deadline`. All three are **terminal**: no bootstrapping past them.
- **Same-step tie:** if the robot reaches the goal and collides in the same step, it counts as a collision.
- **Deadline:** `T = 2L`, where `L` is the shortest-path length on the static map. Reaching the goal at step `T` itself counts as success; `deadline` means `T` steps passed without reaching the goal.
- **Artificial training cutoffs** (if any) are handled separately as truncation.

### 2.5 Reward
- `+1` for reaching the goal.
- `-1` for a collision.
- `-0.5 / T` per step, so running out the full deadline costs at most `-0.5`, less than a collision.
- The step cost also applies on the terminal step. Episode returns: success at step `k` → `1 - 0.5k/T`; collision at step `k` → `-1 - 0.5k/T`; deadline → `-0.5`. So success > deadline > collision always holds.

### 2.6 Observations (identical inputs for every method)
- Static map: shelves and free cells, known globally.
- Robot position and goal position.
- Workers are visible only in a **7×7 window centered on the robot**.
- Each cell is in one of three states: **seen empty / seen worker / not observed**.
  - "Not observed" covers both cells out of view and an outage.
  - A missing reading is never encoded as an empty cell.
- Observation-valid flag: false during an outage.
- Steps since the last valid reading.
- Remaining time: `steps_left / T`.
- Workers are anonymous: no IDs, headings or velocities in the observations.
- *Week-1 decision:* tensor layout (full-map channels vs. a crop centered on the robot), agreed by all three.

### 2.7 Sensor outages
- **Scattered:** isolated single-step drops.
- **Consecutive:** outages of duration **d**.
- Both patterns have the **same overall missing fraction f** (§10).
- *Week-1 decision:* how the schedule is generated. Proposed: the schedule covers the full horizon `T` and drops exactly `round(f·T)` steps. Scattered: drop positions chosen uniformly with no two adjacent. Consecutive: `round(f·T / d)` non-overlapping blocks of `d` steps, at least one visible step between blocks, placed uniformly. Independent per-step drops are not used: they create adjacent drops and a different realized `f` per episode.
  - Feasibility: scattered needs `round(f·T) ≤ ⌊(T+1)/2⌋`; consecutive needs `round(f·T/d)·(d+1) ≤ T+1`. The pilot (§9) picks `f` and `d` within these bounds, and the generator raises an error on infeasible inputs.
  - The two patterns drop `round(f·T)` vs. `d·round(f·T/d)` steps, which can differ by up to about `d/2` steps. Each scenario stores its realized missing fraction, and the report states the largest gap between patterns.
- During an outage the whole 7×7 window is "not observed".

### 2.8 Scenarios and data splits
- Each scenario is **pre-generated**: layout, start, goal, worker initial conditions, worker trajectory seed, outage schedule.
- Workers don't react to the robot, so worker trajectories are fixed per scenario.
- Every controller runs on exactly the same scenarios.
- A scenario must replay identically from its seed.
- **Splits:**
  - train;
  - development (calibration pilot only);
  - validation (Optuna and checkpoint selection);
  - test (final results only).
- Splits use **held-out layouts and held-out worker-trajectory seeds**.

### 2.9 Formal definition (POMDP)
The robot never sees the full state, so the problem is a POMDP `(S, A, P, R, Ω, O, γ)`, not an MDP.
- **State `s_t ∈ S`:** static map, robot position, goal, every worker's position and heading, rule adherence `p` (constant within an episode, varies across episodes), step `t`, and the outage schedule. Worker headings, `p` and the outage schedule are never observed.
- **Actions `A`:** `{N, S, E, W, wait}` (§2.2).
- **Transitions `P(s' | s, a)`:** the robot moves deterministically (§2.2). Workers move by the stochastic rule in §2.3. Outages follow the schedule (§2.7). Collision, success and deadline are absorbing (§2.4).
- **Reward `R(s, a, s')`:** §2.5.
- **Observations `Ω`, `O(o | s')`:** §2.6, deterministic given the state and the outage schedule.
- **Discount `γ`:** one value for all learned methods, set in §10 (proposed 0.99) and not tuned by Optuna, because `γ` defines the objective being compared.
- The ordering success > deadline > collision (§2.5) holds for every `γ ∈ (0, 1]`: with rewards at step `k` discounted by `γ^(k-1)`, success minus deadline is `γ^(k-1) + (0.5/T)·Σ_{i=k}^{T-1} γ^i > 0`, and deadline minus collision is `γ^(k-1) - (0.5/T)·Σ_{i=k}^{T-1} γ^i ≥ γ^(k-1)·(1 - 0.5) > 0`, because the sum has `T - k < T` terms, each at most `γ^(k-1)`.

---

## 3. Methods

All methods get exactly the inputs in §2.6.

| # | Method | What it tests |
|---|---|---|
| 1 | PPO, last 4 frames | Short fixed history |
| 2 | Recurrent PPO (LSTM) | Learned memory |
| 3 | Method 2 + auxiliary occupancy loss (§4) | Added benefit of predictive training |
| 4 | Motion-predicting planner | Learned control vs. explicit tracking and planning |

**Method 4 (motion-predicting planner):**
- Tracks worker motion from the same anonymous observations, and must handle ambiguous tracks.
- Knows the rule's form (constant speed, the turn rule at decision points) but **not** the true test `p`.
- Uses `p̂` fitted on the training mix, the same data the RL agents learn from.
- Predicts occupied cells a few steps ahead (horizon in §10) and plans with **time-expanded A\*, including wait**.

**Why these algorithms** (the guideline grades algorithm selection; this goes into the report's Methodology section):
- The action space is small and discrete and the task is partially observable, so we need a policy that can use memory.
- PPO (Schulman et al., 2017) is a stable, widely used on-policy baseline with single-file reference implementations in CleanRL (Huang et al., 2022), including a recurrent variant.
- Methods 1–3 share the same PPO core and differ only in memory (frame stack vs. LSTM vs. LSTM + auxiliary loss). This isolates the one variable the research question is about.
- A value-based alternative (e.g. DRQN, Hausknecht & Stone, 2015) would need a second core algorithm with its own tuning, which breaks that isolation.
- Auxiliary prediction losses are known to help representation learning under partial observability (Jaderberg et al., 2017), which motivates method 3.

**References** (not main methods):
- **Last-seen A\*:** assumes workers stay where they were last seen; replans every step.
- **Privileged A\*:** sees all current worker positions. It is a privileged-information reference, **not** an upper bound.
- **Freeze heuristic:** waits during an outage, then resumes last-seen A*.

---

## 4. Auxiliary loss (method 3)

- **Target:** occupancy of currently **unobserved, traversable** cells within a **9×9 area centered on the robot**.
  - "Currently unobserved" means not visible at this step: outside the 7×7 window, or any cell during an outage. Past readings don't change the mask.
- **Labels:** simulator ground truth, used **during training only**. It never enters the policy's inputs.
- **Loss:** weighted BCE, with `pos_weight` computed from training data.
  - Masked to the target cells.
  - Skipped when the mask is empty.
- **Outputs:** called **occupancy scores**, not calibrated probabilities.
- **Loss weight λ:** see §10. Setting `λ = 0` recovers method 2 exactly. This is the main ablation.
- **Pairing with method 2:**
  - Initialize the shared network first from the seed.
  - Initialize the auxiliary head from a separate random generator.
  - Use the same random stream for minibatch shuffling.
  - Same seed ⇒ identical starting weights and scenario order. Later divergence (e.g. GPU nondeterminism) is acceptable.

---

## 5. Training

- Train each learned method **once on a mixture** of `d` values, `p` values and the scattered pattern. Don't train a separate model per heatmap cell.
- Use 5 training seeds per learned method (methods 1–3), for 15 main runs. Method 4 and the references are deterministic and run once.
- **Optuna:**
  - Tune and select checkpoints on validation only.
  - Use the same number of trials and the same environment steps per trial for every method.
  - Set a hard tuning budget after the week-2 timing measurement.
- Log wall-clock time for every run.
- Log validation success rate at fixed environment-step intervals during training, for every run (learning curves, §6.1).
- **Stack:** Python, Gymnasium, PyTorch, Optuna, Git.
  - PPO and recurrent PPO adapted from CleanRL.
  - CleanRL's LSTM script is Atari-specific, so its input handling and network need to change.

---

## 6. Evaluation

### 6.1 Navigation
- **Main heatmap:** outage duration `d` × rule adherence `p`, at a fixed missing fraction `f`.
- **Comparison:** the scattered pattern, at the same `f`.
- **Control:** a no-outage run at each `p`, because changing `p` also changes congestion and how often the robot meets workers.
- **Metrics:**
  - success, collision and deadline rates, always reported together;
  - steps to goal, on successful episodes only and always next to success rate.
- **Data efficiency (methods 1–3):** validation success rate vs. environment steps for every seed. Report steps-to-threshold (first evaluation reaching the success threshold in §10) and the area under the curve. Method 2 vs. 3 is paired by seed. This tests the motivation for method 3 (§3); it uses validation scenarios only, never test.

### 6.2 Prediction quality
- Metrics: **average precision and AUROC** on the hidden cells in the 9×9 area. Both are insensitive to calibration.
- **Primary comparison:**
  - Use fixed held-out **observation replays** from one neutral robot trajectory (the freeze-heuristic robot).
  - Feed the identical observation sequence (plus previous actions, if the network takes them) to the planner's tracker and to the method-3 network.
  - Note in the report: the replays are inputs the RL network didn't generate itself, so there is some distribution shift.
- **Secondary:** the same metrics on each method's own trajectories.

### 6.3 Statistics
- Compute per-run means on the identical test set (scenario-level pairing removes scenario noise).
- **Method 2 vs. 3:** pair by seed.
- **Other comparisons between RL methods:** bootstrap over runs. Don't pair seeds across methods.
- **Planner and heuristics** (deterministic):
  - Compute their mean once on the test set.
  - Compare each RL run against that reference, with the bootstrap over RL runs.
- **Heatmap cells with few scenarios:** also resample scenarios inside the bootstrap, keeping the pairing.
- Always show individual seed results next to the CIs.

---

## 7. Team

**Not final (2026-11-12):** roles aren't fixed yet. The table below is a proposed split only. Work is claimed task by task through GitHub issues (see PLAN.md).

| Person | Proposed focus |
|---|---|
| 1 | Environment, last-seen and privileged A\*, freeze heuristic, motion-predicting planner |
| 2 | PPO 4-frame, recurrent PPO, Optuna tuning |
| 3 | From day 1: scenario generator, evaluation scripts, environment tests. Then: method 3 (auxiliary loss), plots |
| All | Observation format and evaluation protocol (week 1); report and presentation. Each person writes their report section as they finish |

---

## 8. Timeline

| Week | Dates | Deliverable |
|---|---|---|
| 1 | Nov 8–14 | **Milestone:** a scenario replays identically across controllers; collision, visibility, outage and deadline tests pass; last-seen and privileged A\* run. Observation format agreed |
| 2 | Nov 15–21 | Related-work list (5–8 papers, for the report's Discussion), motion-predicting planner, freeze heuristic, calibration pilot on development scenarios (§9), PPO solves a small no-outage task, training-time measurement → tuning budget. **Gate:** if anything slips, shrink the warehouse and the number of workers first |
| 3 | Nov 22–28 | Recurrent PPO with outages, first results, environment section of the report |
| 4 | Nov 29–Dec 5 | Method 3 (auxiliary loss) |
| 5 | Dec 6–12 | Tuning + 15 main runs (methods 1–3 × 5 seeds) |
| 6 | Dec 13–19 | Full test evaluation: heatmap, prediction quality, plots |
| 7 | Dec 20–26 | Report, presentation (length to confirm with the lecturer: the guideline says 15 min), demo video (robot during an outage, with vs. without memory) |
| — | Dec 27–31 | Buffer |

**Deliverables:**
- Report: 8–20 pages excluding references, 1.5 line spacing, Arial 12, 1-inch margins.
- Clean, reproducible code in Git.
- Presentation: 10% of the final grade (project: 90%). Length: the guideline says 15 minutes; confirm with the lecturer.

---

## 9. Calibration pilot (week 2)

- Run on **development scenarios only**, using the A\* references and the freeze heuristic. No training is needed.
- **Target 1:** a worker density where privileged A\* succeeds on about 90%+ of scenarios with no outages. This is a pilot target, not a guarantee.
- **Target 2:** a duration range with easy, intermediate and hard conditions, with last-seen A\* going from near its no-outage level to clearly failing.
- **Don't** choose conditions so that the freeze heuristic loses.
- Lock the range in §10 before any final evaluation.

---

## 10. Pilot-determined parameters

| Parameter | Value | Decided on | Reason |
|---|---|---|---|
| Grid size | | | |
| Worker count | | | |
| Worker speed | | | |
| Minimum start–goal distance | | | |
| Missing fraction `f` | | | |
| Duration grid `d` | | | |
| Rule adherence grid `p` | | | |
| Discount `γ` (proposed 0.99) | | | |
| Learning-curve eval interval / success threshold | | | |
| Planner prediction horizon | | | |
| Auxiliary loss weight `λ` | | | |
| `pos_weight` | | | |
| Training steps per run | | | |
| Optuna trials / steps per trial | | | |
| Scenarios per heatmap cell (test) | | | |

---

## 11. Out of scope

- Reward variants.
- Safety-Gymnasium / continuous control.
- Any method not listed in §3.

---

## 12. Change log

| Date | Change | Why |
|---|---|---|
| 2026-11-12 | Lecturer title changed from Dr. to Prof. | Matches the course page |
| 2026-11-12 | Robot speed stated; worker-speed decision added to week 1; collision rule extended for speed > 1 | A worker moving more than 1 cell/step could jump over the robot undetected |
| 2026-11-12 | Algorithm justification added to §3 | The guideline grades algorithm selection |
| 2026-11-12 | Related-work list added to week 2 | The guideline requires comparison to existing work in the report |
| 2026-11-12 | Presentation length marked "to confirm" | Spec said 10 min, guideline says 15 min |
| 2026-11-12 | §2.2 move resolution and §2.5 episode returns stated explicitly | Removes two ways to implement the same rule |
| 2026-11-12 | Outage-schedule generation added as a week-1 decision (§2.7) | Per-step random drops break "isolated" and give different realized `f` |
| 2026-11-12 | §4 auxiliary-loss mask defined; §5 "15 runs" scoped to methods 1–3 | Ambiguous wording |
| 2026-11-12 | §7 marked as a proposal; tasks claimed through GitHub issues | Team hasn't fixed roles yet |
| 2026-11-12 | Goal at step `T` = success (§2.4); outage feasibility bounds and realized-`f` logging (§2.7); collision rules scoped to speed 1 | Edge cases found in review |
| 2026-11-12 | §2.9 formal POMDP definition (state includes `p`) with discount `γ`; data-efficiency learning curves (§5, §6.1); `γ` and curve settings added to §10 | Course lecture 1 (problem formulation) and lecture 11 (data efficiency); `γ` was missing |
