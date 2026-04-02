#!/usr/bin/env python3
"""Plot learning curves for FERMI/OPRO experiments.

Produces three figures:
  1. per_user_<label>.png  — one line per user for a single method
  2. per_user_<label>.png  — same for the other method
  3. comparison.png        — mean ± std for each method on one figure

Usage:
    python -m FERMI.scripts.compare_learning_curves \
        --runs   FERMI/results/fix_10users/20260401-xxx-fermi \
                 FERMI/results/fix_10users/20260401-xxx-opro \
        --labels FERMI OPRO \
        --output_dir FERMI/results/fix_10users/plots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

# 10 distinct colours for per-user lines
USER_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#be185d", "#65a30d", "#ea580c", "#6366f1",
]
# one colour per method for the comparison plot
METHOD_COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]


def load_user_curves(run_dir: Path) -> Dict[str, List[Dict]]:
    path = run_dir / "user_curves.json"
    if not path.exists():
        raise FileNotFoundError(f"user_curves.json not found in {run_dir}")
    with open(path) as f:
        return json.load(f)


def _pad(all_scores: List[List[float]], max_len: int) -> "np.ndarray":
    import numpy as np
    padded = []
    for s in all_scores:
        p = list(s)
        while len(p) < max_len:
            p.append(p[-1] if p else 0.0)
        padded.append(p)
    return np.array(padded)


def plot_per_user(
    run_dir: Path,
    label: str,
    output_path: Path,
    metric_key: str = "best_score",
) -> None:
    """Figure with one line per user (all users of a single method)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    user_curves = load_user_curves(run_dir)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (uid, rows) in enumerate(sorted(user_curves.items())):
        if not rows:
            continue
        scores = [float(r.get(metric_key, 0.0)) for r in rows]
        iters = list(range(len(scores)))
        color = USER_COLORS[i % len(USER_COLORS)]
        ax.plot(iters, scores, color=color, linewidth=1.8,
                label=f"user {uid}", marker="o", markersize=4)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel(metric_key, fontsize=13)
    ax.set_title(f"{label} — per-user learning curves", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_comparison(
    run_dirs: List[Path],
    labels: List[str],
    output_path: Path,
    title: str = "Learning Curve Comparison",
    metric_key: str = "best_score",
) -> None:
    """Figure with mean ± std per method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        user_curves = load_user_curves(run_dir)
        color = METHOD_COLORS[i % len(METHOD_COLORS)]

        all_scores = [
            [float(r.get(metric_key, 0.0)) for r in rows]
            for rows in user_curves.values() if rows
        ]
        if not all_scores:
            print(f"No data for {label}")
            continue

        max_len = max(len(s) for s in all_scores)
        mat = _pad(all_scores, max_len)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        iters = list(range(max_len))
        n = mat.shape[0]

        ax.plot(iters, mean, color=color, linewidth=2.5,
                label=f"{label}  (n={n} users)", marker="o", markersize=5)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel(metric_key, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--title", default="FERMI vs OPRO — LaMP3_rate (10 users)")
    parser.add_argument("--metric_key", default="best_score")
    args = parser.parse_args()

    if len(args.runs) != len(args.labels):
        raise ValueError("--runs and --labels must have the same length")

    run_dirs = [Path(r) for r in args.runs]
    out = Path(args.output_dir)

    # Figure 1 & 2: per-user curves for each method
    for run_dir, label in zip(run_dirs, args.labels):
        plot_per_user(
            run_dir=run_dir,
            label=label,
            output_path=out / f"per_user_{label.lower()}.png",
            metric_key=args.metric_key,
        )

    # Figure 3: comparison (mean ± std)
    plot_comparison(
        run_dirs=run_dirs,
        labels=args.labels,
        output_path=out / "comparison.png",
        title=args.title,
        metric_key=args.metric_key,
    )


if __name__ == "__main__":
    _main()
