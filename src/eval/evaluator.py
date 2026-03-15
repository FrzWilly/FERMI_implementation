from __future__ import annotations

from typing import Dict, List

from FERMI.src.eval.metrics_rate import mae
from FERMI.src.eval.metrics_tag import accuracy
from FERMI.src.eval.metrics_title import mean_rouge_l


class Evaluator:
    def evaluate(self, task: str, method: str, split: str, predictions: List[Dict]) -> Dict:
        evaluable = [p for p in predictions if p.get("gold") is not None]
        n_users = len(set(p.get("user_id") for p in predictions))

        if not evaluable:
            return {
                "task": task,
                "method": method,
                "split": split,
                "primary_metric": self._primary_metric_name(task),
                "metric_value": None,
                "n_samples": len(predictions),
                "n_users": n_users,
            }

        preds = [p["prediction"] for p in evaluable]
        golds = [p["gold"] for p in evaluable]

        if task == "LaMP2_tag":
            metric_name = "accuracy"
            value = accuracy(preds, golds)
        elif task == "LaMP3_rate":
            metric_name = "mae"
            value = mae(preds, golds)
        elif task == "LaMP5_title":
            metric_name = "rouge_l"
            value = mean_rouge_l([str(x) for x in preds], [str(x) for x in golds])
        else:
            raise ValueError(f"Unsupported task: {task}")

        return {
            "task": task,
            "method": method,
            "split": split,
            "primary_metric": metric_name,
            "metric_value": value,
            "n_samples": len(predictions),
            "n_users": n_users,
        }

    @staticmethod
    def _primary_metric_name(task: str) -> str:
        if task == "LaMP2_tag":
            return "accuracy"
        if task == "LaMP3_rate":
            return "mae"
        if task == "LaMP5_title":
            return "rouge_l"
        return "unknown"

