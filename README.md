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
commit	test_acc	memory_gb	status	description
```

`status` can be `keep`, `discard`, or `crash`.
With `--status auto` (default), a run is marked `keep` only if `test_acc` beats the prior best.

## Plot Progress

Generate an autoresearch progress plot from `results.tsv`:

```bash
uv run plot_progress.py --input results.tsv --output progress.png --metric test_acc --higher-is-better --window 0.02
```

For a BPB-style run (lower is better), use:

```bash
uv run plot_progress.py --input results.tsv --output progress.png --metric val_bpb --lower-is-better --window 0.0005
```
