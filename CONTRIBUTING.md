# Contributing

## Workflow

1. Claim the task first: assign yourself its GitHub issue (see [PLAN.md](docs/PLAN.md)).
2. Never push to `main` directly.
3. Create a branch per task: `<name>/<short-topic>`, for example `nell/scenario-generator`.
4. Open a pull request into `main` when the task works and its tests pass.
5. Every PR needs one approving review before merge. Nell reviews and merges.
6. Keep PRs small: one task from [PLAN.md](docs/PLAN.md) per PR.
7. Tick the task in PLAN.md in the same PR.

## Commits

- Commit your own work from your own account. The history is our evidence that everyone contributed.
- Commit often on your branch. Each commit should leave the code working.
- Message: imperative, short, says what changed. Example: `Add swap-collision check to env step`.
- Don't commit results, checkpoints, Optuna databases or notebook outputs (see `.gitignore`).

## Code

- Python 3.10+. Code lives in `src/whnav/`, tests in `tests/`.
- Every function that isn't trivial gets a docstring and type hints.
- Anything random takes an explicit seed or `np.random.Generator`. No global random state.
- Hyperparameters and experiment settings go in `configs/`, not hard-coded.
- Before opening a PR, run:
  ```
  ruff check .
  pytest -q
  ```
  CI runs the same checks on every PR. A red PR isn't reviewed.

## Changing the spec

The spec is frozen (see [SPEC.md](docs/SPEC.md)). Week-1 decisions and §10 parameters are filled in by PR, with the date and the reason. Any other change needs all three to agree and goes into the SPEC change log.
