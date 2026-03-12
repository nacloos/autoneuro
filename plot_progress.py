#!/usr/bin/env python3
"""Plot autoneuro experiment progress from results.tsv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot kept/discarded experiment progress from autoneuro results.tsv"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results.tsv"),
        help="Path to TSV file with columns: commit, test_acc, memory_gb, status, description",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("progress.png"),
        help="Output image path",
    )
    parser.add_argument(
        "--focus-delta",
        type=float,
        default=0.0005,
        help=(
            "Only plot experiments with test_acc >= baseline - focus_delta "
            "(kept points are always shown)"
        ),
    )
    return parser.parse_args()


def read_rows(results_path: Path) -> list[dict]:
    rows = []
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            try:
                acc = float(row["test_acc"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "idx": i,
                    "test_acc": acc,
                    "status": str(row.get("status", "")).strip().lower(),
                    "description": str(row.get("description", "")).strip(),
                }
            )
    return rows


def shorten(text: str, max_len: int = 52) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def main() -> int:
    args = parse_args()
    rows = read_rows(args.results)
    if not rows:
        raise SystemExit(f"No valid rows found in {args.results}")

    baseline = rows[0]["test_acc"]
    threshold = baseline - args.focus_delta

    kept = [r for r in rows if r["status"] == "keep"]
    if not kept:
        raise SystemExit("No rows with status=keep found.")

    visible = [r for r in rows if r["test_acc"] >= threshold or r["status"] == "keep"]
    disc = [r for r in visible if r["status"] != "keep"]
    kept_visible = [r for r in visible if r["status"] == "keep"]

    kept_idx = [r["idx"] for r in kept]
    kept_acc = [r["test_acc"] for r in kept]
    running_best = []
    best_so_far = float("-inf")
    for val in kept_acc:
        best_so_far = max(best_so_far, val)
        running_best.append(best_so_far)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8))

    if disc:
        ax.scatter(
            [r["idx"] for r in disc],
            [r["test_acc"] for r in disc],
            s=18,
            color="#bdbdbd",
            alpha=0.65,
            label="Discarded",
            zorder=1,
        )

    ax.scatter(
        [r["idx"] for r in kept_visible],
        [r["test_acc"] for r in kept_visible],
        s=52,
        facecolor="#2ecc71",
        edgecolor="#196f3d",
        linewidth=1.0,
        label="Kept",
        zorder=3,
    )

    ax.step(
        kept_idx,
        running_best,
        where="post",
        color="#27ae60",
        linewidth=2.0,
        alpha=0.75,
        label="Running best",
        zorder=2,
    )

    for r in kept:
        ax.annotate(
            shorten(r["description"]),
            (r["idx"], r["test_acc"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            color="#2e8b57",
            rotation=28,
            ha="left",
            va="bottom",
        )

    n_total = len(rows)
    n_kept = len(kept)
    ax.set_title(
        f"Autoneuro Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        fontsize=18,
    )
    ax.set_xlabel("Experiment #", fontsize=16)
    ax.set_ylabel("Test Accuracy (higher is better)", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="upper right", fontsize=12)
    ax.grid(True, alpha=0.25)

    y_vals = [r["test_acc"] for r in visible]
    y_min = min(y_vals)
    y_max = max(y_vals)
    margin = max((y_max - y_min) * 0.08, 0.002)
    ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(f"Saved plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
