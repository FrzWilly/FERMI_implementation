from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from FERMI.src.data.split_builder import LAMPSplitBuilder
from FERMI.src.eval.evaluator import Evaluator
from FERMI.src.eval.metrics_title import rouge_l_f1
from FERMI.src.eval.plot_learning_curve import plot_aggregate_learning_curve
from FERMI.src.methods.fermi import FERMIMethod
from FERMI.src.methods.fewshot import FewShotMethod
from FERMI.src.methods.opro import OPROMethod
from FERMI.src.methods.uniform import UniformMethod
from FERMI.src.methods.vanilla import VanillaMethod
from FERMI.src.utils.ids import make_run_id
from FERMI.src.utils.io import ensure_dir, load_config_file, write_json
from FERMI.src.utils.seed import set_seed


def _default_config_paths(task: str, method: str) -> List[Path]:
    task_map = {
        "LaMP2_tag": "lamp2_tag.yaml",
        "LaMP3_rate": "lamp3_rate.yaml",
        "LaMP5_title": "lamp5_title.yaml",
    }
    return [
        Path("FERMI/configs/shared/runtime.yaml"),
        Path("FERMI/configs/shared/models.yaml"),
        Path("FERMI/configs/experiments") / task_map[task],
        Path("FERMI/configs/methods") / f"{method}.yaml",
    ]


def _build_method(method: str, task: str, config: Dict[str, Any]):
    if method == "uniform":
        return UniformMethod(task, config)
    if method == "vanilla":
        return VanillaMethod(task, config)
    if method in {"fewshot_bm25", "fewshot_cont"}:
        cfg = dict(config)
        cfg["retriever"] = "bm25" if method == "fewshot_bm25" else "contriever"
        return FewShotMethod(task, cfg)
    if method == "opro":
        return OPROMethod(task, config)
    if method == "fermi":
        return FERMIMethod(task, config)
    raise ValueError(f"Unsupported method: {method}")


def _enrich_prediction_fields(task: str, record: Dict[str, Any]) -> Dict[str, Any]:
    pred = record.get("prediction")
    gold = record.get("gold")

    is_correct = None
    abs_error = None
    rouge_l = None

    if gold is not None and task == "LaMP2_tag":
        is_correct = str(pred) == str(gold)
    if gold is not None and task == "LaMP3_rate":
        if pred is not None:
            abs_error = abs(float(pred) - float(gold))
    if gold is not None and task == "LaMP5_title":
        rouge_l = rouge_l_f1(str(pred), str(gold))

    record["is_correct"] = is_correct
    record["abs_error"] = abs_error
    record["rouge_l"] = rouge_l
    return record


def _group_by_user(samples: List) -> Dict[str, List]:
    by_user: Dict[str, List] = {}
    for s in samples:
        by_user.setdefault(s.user_id, []).append(s)
    return by_user


def run_per_user(args: argparse.Namespace) -> Dict[str, Any]:
    """Run optimization independently for each user and aggregate results."""
    set_seed(args.seed)

    config: Dict[str, Any] = {}
    if args.config:
        config.update(load_config_file(Path(args.config)))
    else:
        for p in _default_config_paths(args.task, args.method):
            if p.exists():
                config.update(load_config_file(p))

    split_builder = LAMPSplitBuilder(data_root=Path(args.data_root))
    train_samples = split_builder.load_split(args.task, split="train", limit=args.train_limit, check_consistency=True)
    target_samples = split_builder.load_split(args.task, split=args.split, limit=args.limit, check_consistency=True)

    run_id = make_run_id(task=args.task, method=args.method, seed=args.seed)
    run_dir = Path(args.output_dir) / run_id
    users_dir = run_dir / "users"
    ensure_dir(users_dir)

    by_user_train = _group_by_user(train_samples)
    by_user_test = _group_by_user(target_samples)

    all_predictions: List[Dict[str, Any]] = []
    user_curves: Dict[str, List[Dict[str, Any]]] = {}
    user_summaries: Dict[str, Any] = {}

    predictions_path = run_dir / "predictions.json"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"
    user_curves_path = run_dir / "user_curves.json"

    n_users = len(by_user_train)
    for idx, user_id in enumerate(sorted(by_user_train.keys()), 1):
        user_train = by_user_train[user_id]
        user_test = by_user_test.get(user_id, [])

        user_run_dir = users_dir / user_id
        ensure_dir(user_run_dir)

        method = _build_method(args.method, args.task, config)
        method.set_run_context(run_dir=user_run_dir, split=args.split)
        method.fit(user_train)

        # Collect per-user learning curve (only OPRO/FERMI have this).
        curve = method.get_curve_rows() if isinstance(method, OPROMethod) else []
        user_curves[user_id] = curve

        for sample in user_test:
            out = method.predict(sample)
            pred_record = {
                "id": sample.id,
                "task": sample.task,
                "method": args.method,
                "split": args.split,
                "prediction": out.get("prediction"),
                "gold": sample.gold,
                "selected_prompt_id": out.get("selected_prompt_id"),
                "rop_neighbor_ids": out.get("rop_neighbor_ids", []),
                "prediction_source": out.get("prediction_source"),
                "fallback_reason": out.get("fallback_reason"),
                "user_id": sample.user_id,
            }
            all_predictions.append(_enrich_prediction_fields(args.task, pred_record))

        runtime = method.runtime_summary()
        user_summaries[user_id] = {
            "n_train": len(user_train),
            "n_test": len(user_test),
            "n_curve_iters": len(curve),
            "best_prompt_id": runtime.get("best_prompt_id"),
            "best_prompt_score": runtime.get("best_prompt_score"),
        }

        print(f"[{idx}/{n_users}] user={user_id}  train={len(user_train)}  test={len(user_test)}  curve_iters={len(curve)}")

        # Checkpoint after each user so partial results survive interruption.
        evaluator = Evaluator()
        checkpoint_metrics = evaluator.evaluate(
            task=args.task, method=args.method, split=args.split, predictions=all_predictions
        )
        write_json(predictions_path, all_predictions)
        write_json(metrics_path, checkpoint_metrics)
        write_json(user_curves_path, user_curves)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(task=args.task, method=args.method, split=args.split, predictions=all_predictions)

    write_json(predictions_path, all_predictions)
    write_json(metrics_path, metrics)
    write_json(user_curves_path, user_curves)

    # Plot aggregate learning curve with error band.
    lc_png = run_dir / "aggregate_learning_curve.png"
    if any(user_curves.values()):
        plot_aggregate_learning_curve(
            user_curves=user_curves,
            output_path=lc_png,
            title=f"{args.task} · {args.method} · {n_users} users (seed {args.seed})",
        )

    summary = {
        "run_id": run_id,
        "task": args.task,
        "method": args.method,
        "split": args.split,
        "seed": args.seed,
        "mode": "per_user",
        "n_users": n_users,
        "n_predictions": len(all_predictions),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "user_curves_path": str(user_curves_path),
        "aggregate_learning_curve_png": str(lc_png),
        "config": config,
        "user_summaries": user_summaries,
    }
    write_json(summary_path, summary)
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Alias for run_per_user(). All methods always train per user."""
    return run_per_user(args)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one FERMI/LaMP method.")
    parser.add_argument("--task", required=True, choices=["LaMP2_tag", "LaMP3_rate", "LaMP5_title"])
    parser.add_argument(
        "--method",
        required=True,
        choices=["uniform", "vanilla", "fewshot_bm25", "fewshot_cont", "opro", "fermi"],
    )
    parser.add_argument("--config", default=None, help="Optional custom config yaml/json path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--output_dir", default="FERMI/results")
    parser.add_argument("--data_root", default="FERMI/LaMP")
    parser.add_argument("--limit", type=int, default=None, help="Limit target split samples for quick run.")
    parser.add_argument("--train_limit", type=int, default=None, help="Limit training samples for quick run.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_per_user(args)
    print(f"Run completed: {summary['run_id']}")


if __name__ == "__main__":
    main()
