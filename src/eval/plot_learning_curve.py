"""Aggregate per-user learning curves into a single plot with an error band.

Usage (standalone):
    python -m FERMI.src.eval.plot_learning_curve \
        --curves_json path/to/user_curves.json \
        --output path/to/learning_curve.png \
        --title "LaMP3_rate FERMI (50 users)"
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def plot_aggregate_learning_curve(
    user_curves: Dict[str, List[Dict]],
    output_path: Path,
    title: str = "Per-User Learning Curve",
    metric_key: str = "best_score",
    secondary_key: str | None = "best_accuracy",
) -> None:
    """Plot mean ± 1-std error band across users for each iteration.

    Args:
        user_curves: {user_id: [{"iteration": int, "best_score": float, ...}, ...]}
        output_path:  Where to write the .png (or .svg).
        title:        Figure title.
        metric_key:   Primary curve to plot (default "best_score").
        secondary_key: Optional second curve plotted with dashed line.
    """
    output_path = Path(output_path)

    # Collect per-user score arrays.
    all_scores: List[List[float]] = []
    all_secondary: List[List[float]] = []
    for rows in user_curves.values():
        if not rows:
            continue
        scores = [float(r.get(metric_key, 0.0)) for r in rows]
        all_scores.append(scores)
        if secondary_key:
            sec = [float(r.get(secondary_key, 0.0)) for r in rows]
            all_secondary.append(sec)

    if not all_scores:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        _fallback_csv(user_curves, output_path, metric_key, secondary_key)
        return

    max_len = max(len(s) for s in all_scores)

    def _pad(arr: List[List[float]]) -> "np.ndarray":
        padded = []
        for s in arr:
            p = list(s)
            while len(p) < max_len:
                p.append(p[-1] if p else 0.0)
            padded.append(p)
        return np.array(padded)  # (n_users, T)

    mat = _pad(all_scores)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    iters = list(range(max_len))
    n_users = mat.shape[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Primary metric — solid line + shaded band.
    ax.plot(iters, mean, color="#2563eb", linewidth=2.5,
            label=f"mean {metric_key}  (n={n_users} users)")
    ax.fill_between(iters, mean - std, mean + std,
                    color="#2563eb", alpha=0.18, label="±1 std")

    # Secondary metric — dashed line, no band.
    if secondary_key and all_secondary:
        mat2 = _pad(all_secondary)
        mean2 = mat2.mean(axis=0)
        ax.plot(iters, mean2, color="#16a34a", linewidth=2.0,
                linestyle="--", label=f"mean {secondary_key}")

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("Score (0–1)", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks(iters)
    ax.set_xlim(-0.3, max_len - 0.7)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    # Also write companion CSV for reproducibility.
    _write_summary_csv(iters, mean, std, output_path.with_suffix(".csv"), metric_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_summary_csv(
    iters: List[int],
    mean: "np.ndarray",
    std: "np.ndarray",
    path: Path,
    metric_key: str,
) -> None:
    rows = [
        {"iteration": i, f"mean_{metric_key}": float(mean[i]), f"std_{metric_key}": float(std[i])}
        for i in iters
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fallback_csv(
    user_curves: Dict[str, List[Dict]],
    output_path: Path,
    metric_key: str,
    secondary_key: str | None,
) -> None:
    """Write per-user CSV when matplotlib is unavailable."""
    all_scores = {
        uid: [float(r.get(metric_key, 0.0)) for r in rows]
        for uid, rows in user_curves.items()
        if rows
    }
    if not all_scores:
        return
    max_len = max(len(v) for v in all_scores.values())
    csv_path = output_path.with_suffix(".csv")
    user_ids = sorted(all_scores.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration"] + user_ids)
        for i in range(max_len):
            row = [i] + [
                all_scores[uid][i] if i < len(all_scores[uid]) else ""
                for uid in user_ids
            ]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Plot aggregate per-user learning curve.")
    parser.add_argument("--curves_json", required=True,
                        help="JSON file: {user_id: [{iteration, best_score, ...}]}")
    parser.add_argument("--output", required=True, help="Output image path (.png or .svg)")
    parser.add_argument("--title", default="Per-User Learning Curve")
    parser.add_argument("--metric_key", default="best_score")
    parser.add_argument("--secondary_key", default="best_accuracy")
    args = parser.parse_args()

    with open(args.curves_json) as f:
        user_curves = json.load(f)

    plot_aggregate_learning_curve(
        user_curves=user_curves,
        output_path=Path(args.output),
        title=args.title,
        metric_key=args.metric_key,
        secondary_key=args.secondary_key or None,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    _main()
