"""Run a single experiment on a specific GPU with optional model overrides.

Usage:
    CUDA_VISIBLE_DEVICES=2 uv run python run_experiment.py --description "test" --variant base
"""
import argparse
import importlib
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, required=True)
    parser.add_argument("--variant", type=str, default="base",
                        help="Model variant name")
    parser.add_argument("--n-layers", type=int, default=None,
                        help="Override N_LAYERS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override prepare.py settings if needed
    if args.n_layers is not None:
        import prepare
        prepare.N_LAYERS = args.n_layers

    # Now run evaluate
    from evaluate import run_once, _git_commit_short, _auto_status, _append_result, RESULTS_TSV
    commit = _git_commit_short()
    try:
        out = run_once()
        status = _auto_status(RESULTS_TSV, out["test_acc"])
        _append_result(
            RESULTS_TSV,
            commit,
            out["test_acc"],
            out.get("memory_gb", 0.0),
            status,
            args.description,
        )
        print(f"status:     {status}")
        print(f"results:    {RESULTS_TSV}")
    except Exception as exc:
        desc = f"{args.description} | crash: {type(exc).__name__}: {exc}"
        _append_result(RESULTS_TSV, commit, 0.0, 0.0, "crash", desc)
        print("status:     crash")
        print(f"results:    {RESULTS_TSV}")
        raise
