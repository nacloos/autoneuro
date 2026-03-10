# autoneuro

Minimal modular-only extraction for fixed dataset-size experiments.

Scope:
- Model: `modular`
- Task set: `all_yang` (all tasks from `extended_yang19.TASKS`)
- Dataset size: `n_train_samples=500` per task
- Defaults: `shared_weights=True`, `concat_input=True`
- No `analysis.py` dependency
- No `experiment_dataset_size.py` dependency

## Files

- `prepare.py`: fixed experiment constants
- `train.py`: extracted training code (modular-only model registry)
- `model.py`: unified modular architecture + sequence wrapper
- `evaluate.py`: single-run evaluator with summary output and auto-logging

## Run

```bash
uv sync
UV_CACHE_DIR=/storage/nacloos/.uv uv run evaluate.py --description "baseline" > run.log 2>&1
grep "^test_acc:\|^memory_gb:\|^status:\|^results:" run.log
```

## Result Logging

Each run appends one row to `results.tsv` with columns:

```tsv
commit	test_acc	memory_gb	status	description	run_dir
```

`status` can be `keep`, `discard`, or `crash`.
With `--status auto` (default), a run is marked `keep` only if `test_acc` beats the prior best.
