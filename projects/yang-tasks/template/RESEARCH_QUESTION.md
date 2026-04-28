# Research Question

Find a neural architecture that generalizes on the Yang task `dlygointr`.

The training set contains 200 samples from `dlygointr`. The goal is to improve
accuracy on held-out test samples from the same task.

Record experiments and findings in `RESEARCH_LOG.md`.

Run experiments with:

```bash
uv run python train.py
```

Constraints:

- Work only in this workspace.
- Edit only `model.py`.
- Treat `train.py`, `extended_yang19.py`, `pyproject.toml`, and `uv.lock` as
  fixed evaluation harness files.
- Keep dependencies limited to `pyproject.toml`.
- Keep experiments CPU-friendly unless CUDA is available automatically.
