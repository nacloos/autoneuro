"""Run one fixed modular experiment and report metrics.

Usage:
    uv run evaluate.py --description "baseline"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from prepare import (
    BASE_DIR,
    TASK_SET,
    MODEL,
    K,
    N_TRAIN_SAMPLES,
    N_TEST_SAMPLES,
    N_LAYERS,
    SHARED_WEIGHTS,
    CONCAT_INPUT,
    BATCH_SIZE,
    MAX_TRAIN_STEPS,
    N_EPOCHS,
    SEED,
    NO_WANDB,
    RESULTS_DIR,
)
from extended_yang19 import TASKS as NEUROGYM_TASKS
from train import TrainConfig, make_task_config, run_task


RESULTS_TSV = BASE_DIR / "results.tsv"
RESULTS_HEADER_FIELDS = ["commit", "test_acc", "memory_gb", "status", "description", "run_dir"]
RESULTS_HEADER = "\t".join(RESULTS_HEADER_FIELDS) + "\n"
OLD_RESULTS_HEADER = "commit\ttest_acc\tstatus\tdescription\trun_dir"
GPU_POLL_SECONDS = 0.5


def _resolve_tasks() -> list[str]:
    if TASK_SET == "all_yang":
        return list(NEUROGYM_TASKS.keys())
    return [t.strip() for t in TASK_SET.split(",") if t.strip()]


def _query_peak_gpu_memory_mb() -> float | None:
    """Return max used GPU memory (MB) across visible GPUs, or None if unavailable."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None

    vals = []
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            vals.append(float(line))
        except ValueError:
            continue
    if not vals:
        return 0.0
    return max(vals)


def _monitor_peak_gpu_memory(stop_event: threading.Event, peak_box: dict[str, float]) -> None:
    while not stop_event.is_set():
        mb = _query_peak_gpu_memory_mb()
        if mb is not None and mb > peak_box["mb"]:
            peak_box["mb"] = mb
        stop_event.wait(GPU_POLL_SECONDS)


def run_once() -> dict:
    task_names = _resolve_tasks()
    n_tasks = len(task_names)
    total_train_samples = N_TRAIN_SAMPLES * n_tasks
    total_test_samples = N_TEST_SAMPLES * n_tasks

    cfg = TrainConfig(
        task=",".join(task_names),
        multitask=True,
        model=MODEL,
        K=K,
        n_layers=N_LAYERS,
        shared_weights=SHARED_WEIGHTS,
        concat_input=CONCAT_INPUT,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        max_train_steps=MAX_TRAIN_STEPS,
        n_train_samples=total_train_samples,
        n_test_samples=total_test_samples,
        seed=SEED,
        no_wandb=NO_WANDB,
    )

    train_tasks = [make_task_config(name, "train") for name in task_names]
    test_tasks = [make_task_config(name, "test") for name in task_names]

    run_name = f"{TASK_SET}_{MODEL}_{int(time.time())}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    peak_box = {"mb": 0.0}
    monitor = threading.Thread(target=_monitor_peak_gpu_memory, args=(stop_event, peak_box), daemon=True)
    monitor.start()
    try:
        _, accs = run_task(
            train_tasks,
            cfg,
            model_name=MODEL,
            results_dir=run_dir,
            num_workers=cfg.num_workers,
            test_tasks=test_tasks,
        )
    finally:
        stop_event.set()
        monitor.join(timeout=2.0)

    train_acc = float(accs[MODEL]["train"])
    test_acc = float(accs[MODEL]["test"])
    memory_gb = float(peak_box["mb"]) / 1024.0 if peak_box["mb"] > 0 else 0.0

    out = {
        "task_set": TASK_SET,
        "n_tasks": n_tasks,
        "tasks": task_names,
        "model": MODEL,
        "shared_weights": SHARED_WEIGHTS,
        "concat_input": CONCAT_INPUT,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "memory_gb": memory_gb,
        "seed": SEED,
        "n_train_samples_per_task": N_TRAIN_SAMPLES,
        "n_test_samples_per_task": N_TEST_SAMPLES,
        "n_train_samples_total": total_train_samples,
        "n_test_samples_total": total_test_samples,
        "max_train_steps": MAX_TRAIN_STEPS,
        "run_dir": str(run_dir),
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(out, f, indent=2)

    print("---")
    print(f"test_acc:   {test_acc:.6f}")
    print(f"memory_gb:  {memory_gb:.1f}")
    print(f"train_acc:  {train_acc:.6f}")
    print(f"run_dir:    {run_dir}")

    return out


def _git_commit_short() -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        val = proc.stdout.strip()
        return val if val else "nogit"
    except Exception:
        return "nogit"


def _ensure_results_tsv(path: Path) -> None:
    if not path.exists():
        path.write_text(RESULTS_HEADER)
        return

    lines = path.read_text().splitlines()
    if not lines:
        path.write_text(RESULTS_HEADER)
        return

    header = lines[0].strip()
    if header == "\t".join(RESULTS_HEADER_FIELDS):
        return

    if header == OLD_RESULTS_HEADER:
        migrated = [RESULTS_HEADER.rstrip("\n")]
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            commit, test_acc, status, description = parts[:4]
            run_dir = "\t".join(parts[4:])
            migrated.append(f"{commit}\t{test_acc}\t0.0\t{status}\t{description}\t{run_dir}")
        path.write_text("\n".join(migrated) + "\n")
        return

    raise RuntimeError(f"Unexpected results.tsv header: {header}")


def _parse_best_test_acc(path: Path) -> float | None:
    if not path.exists():
        return None

    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            i_test_acc = header.index("test_acc")
            i_status = header.index("status")
        except ValueError:
            return None

        best = None
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if max(i_test_acc, i_status) >= len(parts):
                continue
            status = parts[i_status].strip().lower()
            if status == "crash":
                continue
            try:
                val = float(parts[i_test_acc])
            except Exception:
                continue
            if best is None or val > best:
                best = val
        return best


def _auto_status(path: Path, test_acc: float) -> str:
    best = _parse_best_test_acc(path)
    if best is None:
        return "keep"
    return "keep" if test_acc > best else "discard"


def _clean_description(text: str) -> str:
    return text.replace("\t", " ").replace("\n", " ").strip()


def _append_result(
    path: Path,
    commit: str,
    test_acc: float,
    memory_gb: float,
    status: str,
    description: str,
    run_dir: str,
) -> None:
    _ensure_results_tsv(path)
    with path.open("a") as f:
        f.write(
            f"{commit}\t{test_acc:.6f}\t{memory_gb:.1f}\t{status}\t{_clean_description(description)}\t{run_dir}\n"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed autoneuro experiment and log results")
    parser.add_argument(
        "--description",
        type=str,
        default="baseline",
        help="Short description to record in results.tsv",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="auto",
        choices=["auto", "keep", "discard"],
        help="Result status. auto=keep iff test_acc beats prior best",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip results.tsv logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    commit = _git_commit_short()
    run_dir = ""
    try:
        out = run_once()
        run_dir = str(out.get("run_dir", ""))
        if not args.no_log:
            status = args.status if args.status != "auto" else _auto_status(RESULTS_TSV, out["test_acc"])
            _append_result(
                RESULTS_TSV,
                commit,
                out["test_acc"],
                out.get("memory_gb", 0.0),
                status,
                args.description,
                run_dir,
            )
            print(f"status:     {status}")
            print(f"results:    {RESULTS_TSV}")
    except Exception as exc:
        if not args.no_log:
            desc = f"{args.description} | crash: {type(exc).__name__}: {exc}"
            _append_result(RESULTS_TSV, commit, 0.0, 0.0, "crash", desc, run_dir)
            print("status:     crash")
            print(f"results:    {RESULTS_TSV}")
        raise
