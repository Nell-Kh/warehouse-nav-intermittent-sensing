# SPEC — Warehouse Navigation under Intermittent Sensing

RL course project (Prof. Teddy Lazebnik, University of Haifa). Group of 3.

**Status:** research scope and methods frozen on 2026-11-05. The only open items are the parameters in §10. The pilot sets the parameters, and they're recorded there with the date and the reason. Don't add methods. Every change to this file is logged in §12.

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
- Moving into a shelf, a wall or off the map leaves the robot in place (decided 2026-11-25).

### 2.3 Workers
- **Speed:** constant between decision points (§10). In aisles, workers always move forward.
- Worker speed is 1 cell per step (decided 2026-11-25). With speed > 1 a worker can cross the robot's path inside one step, which the two rules in §2.4 don't catch; choosing speed > 1 therefore also requires defining a sub-step collision check before implementation.
- **Decision points** (intersections, corners, dead ends):
  - with probability **p**, follow the rule **straight → right → left → back**, taking the first legal move;
  - otherwise, pick uniformly among legal moves, excluding back unless it's a dead end.
  - Precisely: a cell is a decision point unless straight ahead is the only legal move other than back (an aisle). In an aisle the worker moves forward and uses no randomness. In areas wider than one cell, every cell with a side opening is a decision point.
- **Randomness:** workers draw only from their own generator, seeded by the scenario's worker-trajectory seed (§2.8). Workers are processed in list order; at a decision point a worker draws once to choose rule vs. random, and once more for the random choice. Nothing else draws from this generator, so trajectories are identical for every controller.
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
- Tensor layout: full-map channels (decided 2026-11-25):
  - `grid`, shape `(6, H, W)`: blocked, robot, goal, seen empty, seen worker, not observed. Every free cell has exactly one of the last three set; blocked cells have none.
  - `scalars`, shape `(3,)`: valid flag, steps since the last valid reading divided by `T`, `steps_left / T`.
  - Why full-map: the grid size is fixed and the map is known, so the goal is visible directly and no separate goal-direction input is needed.
- The window is not blocked by shelves, and it is clipped at the map edge.
- If no reading has been valid yet, "steps since the last valid reading" counts from the start of the episode (1 at the first observation).

### 2.7 Sensor outages
- **Scattered:** isolated single-step drops.
- **Consecutive:** outages of duration **d**.
- Both patterns have the **same expected missing fraction f** (§10).
- Schedule generation (decided 2026-11-25): the schedule covers the full horizon `T`. Scattered is the case `d = 1`. The number of blocks is `x = f·T / d` rounded at random: up with probability equal to its fractional part, down otherwise. The blocks (exactly `d` steps each, at least one visible step between any two) are placed uniformly over all valid placements. Independent per-step drops are not used: they create adjacent drops.
  - Why random rounding: with ordinary rounding, the realized missing fraction of the consecutive pattern drifts with `d` (e.g. 0.40 instead of 0.30 at `T = 20`, `d = 4`), so the `d` axis of the heatmap would be confounded with `f`. Random rounding makes the expected missing fraction exactly `f` for every `d`; each single schedule is off by fewer than `d` steps.
  - Feasibility: when `f > 0`, at least one block is required (`f·T / d ≥ 1`), so an outage condition never silently becomes a no-outage episode; and `⌈f·T/d⌉·(d+1) ≤ T+1`. The generator raises an error otherwise. So the largest `d` must satisfy `d ≤ f·T_min`, where `T_min = 2 ×` the minimum start–goal distance; the pilot (§9) picks `f`, the `d` grid and the minimum distance together.
  - Entry `i` of the schedule (0-based) applies to the observation the robot uses for its `(i+1)`-th action, so the first observation can be missing too. Each scenario stores its realized missing fraction.
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
- Worker start positions exclude the robot's start and every cell within Manhattan distance 2 of it, so no episode begins with an unavoidable collision (decided 2026-11-25).
- The training mix includes 10% no-outage episodes, so the no-outage control (§6.1) is not out of distribution (decided 2026-11-25).

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
  - Set a hard tuning budget after the training-time measurement (RL-3).
- The policy network gets only the observation in §2.6, not the previous action (decided 2026-11-25).
- At evaluation, learned methods sample their action from the policy, with a generator seeded per scenario so every run is reproducible (decided 2026-11-25). Sampling, not argmax: without full memory, a deterministic policy can get stuck in loops.
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

Three lanes (decided 2026-11-25). Each person owns one end-to-end piece and its interface; tasks are GitHub issues with the lane prefix.

| Lane | Owner | Owns |
|---|---|---|
| Simulator and references (`ENV-*`) | Nell | Environment and Gymnasium API, layouts, scenarios and splits, last-seen and privileged A\*, freeze heuristic, calibration pilot (§9, §10), the planner's `p̂`, prediction quality (§6.2), rendering and demo video |
| Learned methods (`RL-*`) | Teammate B | PPO core, methods 1–3, auxiliary target, learning curves and data efficiency (§6.1), Optuna tuning, the 15 main runs |
| Planner and evaluation (`EVAL-*`) | Teammate C | Controller interface, episode runner, results format, method 4, navigation metrics, statistics (§6.3), main figures, final evaluation |
| All (`TEAM-*`) | Everyone | Related work, report, presentation, reproducibility check. Each person writes their report sections as their part works |

---

## 8. Timeline

Replaced on 2026-11-25 (the original weeks 1–3 slipped). Dates are due dates for merged PRs; the GitHub issues hold the detail.

| Dates | Simulator and references (Nell) | Learned methods (B) | Planner and evaluation (C) | Everyone |
|---|---|---|---|---|
| Nov 26 – Dec 5 | ENV-1 environment (Dec 1); ENV-2 layouts (Dec 3); ENV-3 scenarios and splits, ENV-4 replay tests (Dec 4) | RL-1 PPO core (Dec 2); RL-6 auxiliary target (Dec 3) | EVAL-1/2 runner and results format (Dec 2); EVAL-3 metrics (Dec 5) | — |
| Dec 6 – 12 | ENV-5/6 references (Dec 6); ENV-9 `p̂` (Dec 7); ENV-7 pilot, lock §10 (Dec 9); environment figures (Dec 12) | RL-2 method 1 (Dec 6); RL-3 timing and budget (Dec 7); RL-4 method 2 (Dec 9); RL-5 controller and curves (Dec 10); RL-7 method 3 (Dec 12) | EVAL-6 tracker (Dec 9); EVAL-7 planner (Dec 12) | Related work (Dec 10) |
| Dec 13 – 19 | ENV-10 prediction quality (Dec 17); report draft (Dec 18) | RL-8 tuning (Dec 15); RL-9 15 runs (Dec 19) | EVAL-4/5 statistics and figures, EVAL-8 dry run (Dec 15) | Machines shared for training |
| Dec 20 – 26 | Report sections (Dec 22); demo video (Dec 23) | RL-10 data efficiency (Dec 20); RL-11 report (Dec 23) | EVAL-8 final evaluation (Dec 21); EVAL-9 report (Dec 23) | Report and presentation (Dec 26) |
| Dec 27 – 31 | Buffer | Buffer | Buffer | Fresh-clone reproducibility check (Dec 27) |

**Checkpoints:** Dec 2 (PPO core and episode runner merged) and Dec 9 (the planner's tracker passes its tests). If one is missed, the lane's work is redistributed that week.
**Gate:** if anything slips, shrink the warehouse and the number of workers first. If time runs short later, cut in this order: fewer Optuna trials, then the secondary prediction metrics, then fewer `d × p` cells. Keep the four methods and the five seeds.

**Deliverables:**
- Report: 8–20 pages excluding references, 1.5 line spacing, Arial 12, 1-inch margins.
- Clean, reproducible code in Git.
- Presentation: 15 minutes, as the guideline states; 10% of the final grade (project: 90%).

---

## 9. Calibration pilot (ENV-7)

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

**Provisional values** for development until the pilot locks the table above (ENV-7, Dec 9): 15 × 15 grid including the outer wall; 6 workers; minimum start–goal distance 14 (so `T ≥ 28`); `f = 0.3`; `d ∈ {2, 4, 6, 8}`; `p ∈ {0.5, 0.7, 0.9, 1.0}`; `γ = 0.99`; 100 test scenarios per heatmap cell. The minimum distance is 14 rather than 10 because `d = 8` needs `f·T ≥ 8` (§2.7).

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
| 2026-11-19 | §2.3 decision point defined precisely; worker random-draw order fixed | Needed to implement workers reproducibly |
| 2026-11-25 | §2.7 rounding rule, schedule indexing, uniform placement and the zero-block error stated | Needed to implement outage schedules without ambiguity |
| 2026-11-25 | §2.6 observation layout written down (full-map channels, pending team confirmation); no occlusion; counter start convention | Needed to implement observations |
| 2026-11-25 | Week-1 decisions recorded (§2.2, §2.3, §2.6); worker start exclusion and 10% no-outage training episodes (§2.8); no previous-action input and sampled evaluation actions (§5) | Open decisions left to the environment owner at the Nov 25 team meeting |
| 2026-11-25 | §2.7 block count rounded at random instead of half up; feasibility needs at least one block | Ordinary rounding made the realized missing fraction depend on `d`, confounding the heatmap |
| 2026-11-25 | §7 three lanes; §8 timeline replaced; presentation set to 15 minutes; provisional parameters under §10 | Work plan agreed at the Nov 25 team meeting |
