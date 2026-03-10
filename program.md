# autoneuro

This is an autoresearch-style experiment loop adapted to the fixed autoneuro setup.

## Setup

To start a run, do the same setup style as `autoresearch`:

1. Confirm fixed experiment config in `prepare.py` (do not modify):
   - Task set: `all_yang`
   - Model family: `modular`
   - Dataset size: `500` train + `500` test per task
   - Shared weights + concat input enabled
2. Read in-scope files:
   - `prepare.py` (fixed constants + run config)
   - `train.py` (training loop, optimization, plotting)
   - `model.py` (modular architecture)
   - `evaluate.py` (fixed evaluator + logging harness)
3. Initialize `results.tsv` if missing (header is auto-created by `evaluate.py`).

## Experimentation

Each experiment is one full fixed run:

`UV_CACHE_DIR=/storage/nacloos/.uv uv run evaluate.py --description "<what changed>" > run.log 2>&1`

What you CAN change:
- `train.py`
- `model.py`

What you CANNOT change:
- `prepare.py`
- Fixed task/model/data budget encoded in `prepare.py`

Goal:
- Maximize `test_acc` (higher is better)

## Output format

`evaluate.py` prints summary lines like:

- `test_acc: ...`
- `memory_gb: ...`
- `train_acc: ...`
- `run_dir: ...`
- `status: ...`
- `results: ...`

Useful extraction:

`grep "^test_acc:\|^memory_gb:\|^status:\|^results:" run.log`

## Logging results

`evaluate.py` appends one row per run to `results.tsv`:

`commit  test_acc  memory_gb  status  description  run_dir`

Status is `keep`, `discard`, or `crash`.
With `--status auto` (default), `keep` means `test_acc` beats prior best non-crash run.

## Experiment loop (same style as autoresearch)

LOOP:

1. Check current git branch/commit.
2. Edit `train.py` and/or `model.py` with one idea.
3. Commit the change.
4. Run:
   `UV_CACHE_DIR=/storage/nacloos/.uv uv run evaluate.py --description "<idea>" > run.log 2>&1`
5. Read:
   `grep "^test_acc:\|^memory_gb:\|^status:\|^results:" run.log`
6. If crash, inspect with:
   `tail -n 50 run.log`
7. Keep/discard decision:
   - If improved (`status: keep`), stay on current commit.
   - If not improved (`status: discard`), reset to pre-experiment commit.

If this directory is not in a git repo, you can still run experiments, but commit/rollback loop control is unavailable.
