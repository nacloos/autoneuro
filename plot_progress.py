#!/usr/bin/env python3
"""Plot autoresearch progress from results.tsv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_status(status: str) -> str:
    return (status or "").strip().lower()


def _read_results(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"Missing TSV header in {path}")
        rows = []
        for i, row in enumerate(reader):
            clean = {k.strip(): (v or "").strip() for k, v in row.items() if k is not None}
            clean["exp_idx"] = str(i)
            rows.append(clean)
    return list(reader.fieldnames), rows


def _infer_metric(columns: List[str], rows: List[Dict[str, str]], metric: Optional[str]) -> str:
    if metric:
        if metric not in columns:
            raise ValueError(f"--metric '{metric}' not found in TSV columns: {columns}")
        return metric

    preferred = ["val_bpb", "validation_bpb", "test_acc", "val_loss", "loss"]
    for name in preferred:
        if name in columns:
            return name

    ignored = {"commit", "status", "description", "memory_gb"}
    for col in columns:
        if col in ignored:
            continue
        if any(_parse_float(r.get(col, "")) is not None for r in rows):
            return col

    raise ValueError("Could not infer a numeric metric column; pass --metric explicitly.")


def _infer_lower_is_better(metric: str, lower_is_better: bool, higher_is_better: bool) -> bool:
    if lower_is_better and higher_is_better:
        raise ValueError("Use at most one of --lower-is-better or --higher-is-better.")
    if lower_is_better:
        return True
    if higher_is_better:
        return False
    metric_lower = metric.lower()
    return any(token in metric_lower for token in ("bpb", "loss", "error", "wer", "perplexity"))


def _metric_label(metric: str) -> str:
    if metric == "val_bpb":
        return "Validation BPB"
    if metric == "test_acc":
        return "Test Accuracy"
    return metric.replace("_", " ").title()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results.tsv"), help="Path to results TSV.")
    parser.add_argument("--output", type=Path, default=Path("progress.png"), help="Output PNG path.")
    parser.add_argument("--metric", type=str, default=None, help="Metric column to plot.")
    parser.add_argument("--window", type=float, default=None, help="Optional focus window around baseline.")
    parser.add_argument("--lower-is-better", action="store_true", help="Treat lower metric values as better.")
    parser.add_argument("--higher-is-better", action="store_true", help="Treat higher metric values as better.")
    parser.add_argument("--max-label-len", type=int, default=45, help="Max kept-label length before truncation.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG DPI.")
    args = parser.parse_args()

    columns, rows = _read_results(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    metric = _infer_metric(columns, rows, args.metric)
    lower_is_better = _infer_lower_is_better(metric, args.lower_is_better, args.higher_is_better)

    valid = []
    keep_count = 0
    for row in rows:
        status = _normalize_status(row.get("status", ""))
        value = _parse_float(row.get(metric, ""))
        if status == "keep":
            keep_count += 1
        if status == "crash" or value is None:
            continue
        valid.append(
            {
                "exp_idx": int(row["exp_idx"]),
                "status": status,
                "metric": value,
                "description": row.get("description", ""),
            }
        )

    if not valid:
        raise ValueError("No plottable rows (non-crash rows with numeric metric).")

    baseline = valid[0]["metric"]

    if args.window is None:
        focus = valid
    elif lower_is_better:
        focus = [r for r in valid if r["metric"] <= baseline + args.window]
    else:
        focus = [r for r in valid if r["metric"] >= baseline - args.window]
    if not focus:
        focus = valid

    discarded = [r for r in focus if r["status"] != "keep"]
    kept_focus = [r for r in focus if r["status"] == "keep"]
    kept_all = [r for r in valid if r["status"] == "keep"]

    fig, ax = plt.subplots(figsize=(16, 8))

    if discarded:
        ax.scatter(
            [r["exp_idx"] for r in discarded],
            [r["metric"] for r in discarded],
            c="#cccccc",
            s=14,
            alpha=0.55,
            zorder=2,
            label="Discarded",
        )

    if kept_focus:
        ax.scatter(
            [r["exp_idx"] for r in kept_focus],
            [r["metric"] for r in kept_focus],
            c="#2ecc71",
            s=52,
            zorder=4,
            edgecolors="black",
            linewidths=0.5,
            label="Kept",
        )

    if kept_all:
        running = []
        best = None
        for r in kept_all:
            if best is None:
                best = r["metric"]
            elif lower_is_better:
                best = min(best, r["metric"])
            else:
                best = max(best, r["metric"])
            running.append(best)
        ax.step(
            [r["exp_idx"] for r in kept_all],
            running,
            where="post",
            color="#27ae60",
            linewidth=2.0,
            alpha=0.75,
            zorder=3,
            label="Running best",
        )

    for r in kept_focus:
        desc = (r["description"] or "").strip()
        if len(desc) > args.max_label_len:
            desc = desc[: args.max_label_len - 3] + "..."
        ax.annotate(
            desc,
            (r["exp_idx"], r["metric"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
            color="#1a7a3a",
            alpha=0.9,
            rotation=30,
            ha="left",
            va="bottom",
        )

    y_values = [r["metric"] for r in focus]
    y_min = min(y_values)
    y_max = max(y_values)
    y_span = max(y_max - y_min, 1e-9)
    y_margin = y_span * 0.12
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.set_xlabel("Experiment #", fontsize=12)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_ylabel(f"{_metric_label(metric)} ({direction})", fontsize=12)
    ax.set_title(
        f"Autoresearch Progress: {len(rows)} Experiments, {keep_count} Kept Improvements",
        fontsize=17,
    )
    ax.set_xlim(-1, max(r["exp_idx"] for r in valid) + 1)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Metric: {metric} (lower_is_better={lower_is_better})")
    print(f"Plotted {len(focus)} non-crash rows ({len(kept_focus)} kept in focus)")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
