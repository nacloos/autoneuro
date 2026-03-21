# Autoresearch: Yang Tasks Test Accuracy

## Objective
Maximize mean test accuracy across all 93 extended Yang19 neurogym tasks. Each task gets 200 training trials. A single multi-task RNN is trained with task identity (one-hot rule input) to perform all tasks simultaneously. The tasks span 4 families: Go (reach/anti-reach), DM (decision-making), DlyDM (delayed DM), and Match (delay match-to-sample).

## Metrics
- **Primary**: mean_acc (unitless 0-1, higher is better) — mean per-task test accuracy during decision periods
- **Secondary**: go_acc, dm_acc, dlydm_acc, match_acc — per-family averages; tasks_above_50, tasks_above_80 — count of tasks exceeding thresholds

## How to Run
`./autoresearch.sh` — outputs `METRIC name=number` lines.

## Files in Scope
- `train.py` — training script: data generation, model, training loop, evaluation. **Primary optimization target.**
- `extended_yang19.py` — task definitions (READ ONLY, do not modify)

## Off Limits
- `extended_yang19.py` — task environments, must not be changed
- `pyproject.toml` / `uv.lock` — no new dependencies

## Constraints
- 200 training trials per task (NUM_TRAIN_TRIALS=200)
- No new dependencies beyond torch, numpy, neurogym, matplotlib, gymnasium
- Must work on CPU (CUDA optional)
- Script must complete in < 10 minutes

## What's Been Tried
(Will be updated as experiments accumulate)
