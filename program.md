# autoneuro

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar10`). The branch `autoneuro/<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b autoneuro/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - fixed constants and evaluation setup. Do not modify.
   - `train.py` - training loop and optimization logic.
   - `model.py` - modular architecture definition.
   - `evaluate.py` - fixed experiment harness and metric logging.
4. **Verify environment**: confirm runs use `UV_CACHE_DIR=/storage/nacloos/.uv`.
5. **Initialize results.tsv**: create `results.tsv` with just the header row if missing. The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The run uses a **fixed setup** from `prepare.py` (task set, model family, sample counts, layer count, and training budget). You launch it as:

`UV_CACHE_DIR=/storage/nacloos/.uv uv run evaluate.py --description "<what changed>"`

**What you CAN do:**
- Modify `train.py` and `model.py`. Everything in those files is fair game: architecture details, optimizer, hyperparameters, routing behavior, regularization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only and defines fixed task/model/data constraints.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation target metric. `test_acc` from `evaluate.py` is the ground truth metric.

**The goal is simple: get the highest test_acc.** Since setup is fixed, compare ideas on that metric directly.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful `test_acc` gains, but it should not blow up dramatically.

**Simplicity criterion**: all else being equal, simpler is better. A tiny gain with heavy complexity is usually not worth keeping. Equal or better performance with less complexity is a win.

**The first run**: your first run should establish baseline, so run as-is first.

## Output format

Once the script finishes it prints summary lines like:

```
---
test_acc:   0.555932
memory_gb:  7.8
train_acc:  0.997452
run_dir:    /storage/nacloos/projects/autoneuro/results/all_yang_modular_...
status:     keep
results:    /storage/nacloos/projects/autoneuro/results.tsv
```

Extract key lines from the log file:

```
grep "^test_acc:\|^memory_gb:\|^status:\|^results:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated - commas break descriptions).

The TSV has a header row and 5 columns:

```
commit	test_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. test_acc achieved (e.g. 0.555932) - use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 7.8) - use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	test_acc	memory_gb	status	description
a1b2c3d	0.555932	7.8	keep	baseline
b2c3d4e	0.561200	8.0	keep	raise learning rate
c3d4e5f	0.550100	7.9	discard	swap activation
d4e5f6g	0.000000	0.0	crash	double width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoneuro/mar10` or `autoneuro/mar10-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` and/or `model.py` with one experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `UV_CACHE_DIR=/storage/nacloos/.uv uv run evaluate.py --description "<idea>" > run.log 2>&1` (redirect everything - do NOT use tee or let output flood your context).
5. Read out the results: `grep "^test_acc:\|^memory_gb:\|^status:" run.log`.
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit `results.tsv`, leave it untracked by git).
8. If test_acc improved (higher), you "advance" the branch, keeping the git commit.
9. If test_acc is equal or worse, you git reset back to where you started.

The idea is that you are an autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: if a run takes unusually long for this fixed setup, treat it as a failure (discard and revert).

**Crashes**: if a run crashes (OOM or bug), use judgment. If it is trivial to fix, fix and rerun. If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP**: once the experiment loop begins, do not pause to ask whether to continue. Keep running experiments until the human interrupts you.
