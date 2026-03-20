from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from FERMI.src.data.split_builder import LAMPSplitBuilder
from FERMI.src.eval.evaluator import Evaluator
from FERMI.src.eval.metrics_title import rouge_l_f1
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
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
    ensure_dir(run_dir)

    method = _build_method(args.method, args.task, config)
    method.set_run_context(run_dir=run_dir, split=args.split)
    method.fit(train_samples)

    predictions: List[Dict[str, Any]] = []
    for sample in target_samples:
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
        predictions.append(_enrich_prediction_fields(args.task, pred_record))

    evaluator = Evaluator()
    metrics = evaluator.evaluate(task=args.task, method=args.method, split=args.split, predictions=predictions)

    predictions_path = run_dir / "predictions.json"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"

    summary = {
        "run_id": run_id,
        "task": args.task,
        "method": args.method,
        "split": args.split,
        "seed": args.seed,
        "n_predictions": len(predictions),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "config": config,
        "artifacts": method.artifact_summary(),
        "method_runtime": method.runtime_summary(),
    }

    write_json(predictions_path, predictions)
    write_json(metrics_path, metrics)
    write_json(summary_path, summary)
    return summary


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
    summary = run(args)
    print(f"Run completed: {summary['run_id']}")


if __name__ == "__main__":
    main()
