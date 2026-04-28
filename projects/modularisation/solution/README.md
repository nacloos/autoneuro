# Modularisation via Noise

This folder contains a script version of the original notebook experiment.

## Run

From this directory:

```bash
uv run python run.py
```

The script trains all configured settings and saves outputs under:

```text
results/<setting>/
```

Each setting writes:

- `model.pth`
- `losses.npy`
- `losses.pdf`
- `scatter.pdf`
- `spectrum.pdf`
- `connectivity.pdf`

## Notes

- Dependencies are listed in `pyproject.toml`.
- `torch` is configured to install from the CPU-only PyTorch wheel index.
- The experiment generates its datasets in memory; no external data files are required.
- Training is configured for `10000` epochs per setting, so a full run can take a while.
