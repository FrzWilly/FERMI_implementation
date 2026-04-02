from __future__ import annotations

import errno
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from FERMI.src.utils.io import append_jsonl, ensure_dir, write_csv, write_json


def _trim_failure_cases(records: Sequence[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    trimmed: List[Dict[str, Any]] = []
    for record in list(records)[: max(1, limit)]:
        trimmed.append(
            {
                "index": record.get("index"),
                "id": record.get("id"),
                "question": record.get("question"),
                "gold": record.get("gold"),
                "prediction": record.get("prediction"),
                "score": record.get("score"),
            }
        )
    return trimmed


def _snapshot_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": entry.get("prompt_id"),
        "iteration": entry.get("iteration"),
        "score": float(entry.get("score", 0.0)),
        "accuracy": float(entry.get("accuracy", 0.0)),
        "loss": float(entry.get("loss", 0.0)),
        "misaligned_count": int(entry.get("misaligned_count", 0)),
        "prompt_text": entry.get("text", ""),
        "failure_cases": _trim_failure_cases(entry.get("misaligned_records", [])),
    }


def _line_path(values: List[float], width: int, height: int, left: int, top: int, right: int, bottom: int) -> str:
    if not values:
        return ""
    plot_width = max(1, width - left - right)
    plot_height = max(1, height - top - bottom)
    max_x = max(1, len(values) - 1)

    def _xy(index: int, value: float) -> tuple[float, float]:
        x = left + (plot_width * index / max_x)
        y = top + ((1.0 - min(max(value, 0.0), 1.0)) * plot_height)
        return x, y

    coords = [_xy(idx, value) for idx, value in enumerate(values)]
    head_x, head_y = coords[0]
    parts = [f"M {head_x:.2f} {head_y:.2f}"]
    for x, y in coords[1:]:
        parts.append(f"L {x:.2f} {y:.2f}")
    return " ".join(parts)


def build_learning_curve_svg(curve_rows: Sequence[Dict[str, Any]]) -> str:
    width = 960
    height = 540
    left = 70
    right = 40
    top = 40
    bottom = 70

    score_values = [float(row.get("best_score", 0.0)) for row in curve_rows]
    accuracy_values = [float(row.get("best_accuracy", 0.0)) for row in curve_rows]
    loss_values = [float(row.get("best_loss", 0.0)) for row in curve_rows]

    score_path = _line_path(score_values, width, height, left, top, right, bottom)
    accuracy_path = _line_path(accuracy_values, width, height, left, top, right, bottom)
    loss_path = _line_path(loss_values, width, height, left, top, right, bottom)

    x_ticks: List[str] = []
    plot_width = max(1, width - left - right)
    plot_height = max(1, height - top - bottom)
    tick_count = max(1, len(curve_rows))
    for idx, row in enumerate(curve_rows):
        x = left + (plot_width * idx / max(1, tick_count - 1))
        label = str(row.get("iteration", idx))
        x_ticks.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#666" stroke-width="1"/>')
        x_ticks.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 24}" font-size="12" text-anchor="middle" fill="#333">{label}</text>'
        )

    y_ticks: List[str] = []
    for step in range(6):
        value = step / 5.0
        y = top + ((1.0 - value) * plot_height)
        y_ticks.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#666" stroke-width="1"/>')
        y_ticks.append(f'<text x="{left - 12}" y="{y + 4:.2f}" font-size="12" text-anchor="end" fill="#333">{value:.1f}</text>')
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width / 2:.2f}" y="24" font-size="20" text-anchor="middle" fill="#111827">Learning Curve</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="2"/>
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="2"/>
  {''.join(y_ticks)}
  {''.join(x_ticks)}
  <path d="{score_path}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <path d="{accuracy_path}" fill="none" stroke="#16a34a" stroke-width="3"/>
  <path d="{loss_path}" fill="none" stroke="#dc2626" stroke-width="3"/>
  <text x="{width / 2:.2f}" y="{height - 18}" font-size="14" text-anchor="middle" fill="#111827">iteration</text>
  <text x="24" y="{height / 2:.2f}" font-size="14" text-anchor="middle" fill="#111827" transform="rotate(-90, 24, {height / 2:.2f})">value</text>
  <rect x="{width - 240}" y="50" width="170" height="78" rx="8" fill="#f9fafb" stroke="#d1d5db"/>
  <line x1="{width - 225}" y1="72" x2="{width - 195}" y2="72" stroke="#2563eb" stroke-width="3"/>
  <text x="{width - 185}" y="76" font-size="12" fill="#111827">score</text>
  <line x1="{width - 225}" y1="94" x2="{width - 195}" y2="94" stroke="#16a34a" stroke-width="3"/>
  <text x="{width - 185}" y="98" font-size="12" fill="#111827">accuracy</text>
  <line x1="{width - 225}" y1="116" x2="{width - 195}" y2="116" stroke="#dc2626" stroke-width="3"/>
  <text x="{width - 185}" y="120" font-size="12" fill="#111827">loss</text>
</svg>'''


class TrainingArtifacts:
    def __init__(self, run_dir: Path, method_name: str) -> None:
        self.run_dir = run_dir
        self.method_name = method_name
        self.training_trace_jsonl_path = run_dir / "training_trace.jsonl"
        self.training_trace_csv_path = run_dir / "training_trace.csv"
        self.learning_curve_csv_path = run_dir / "learning_curve.csv"
        self.learning_curve_svg_path = run_dir / "learning_curve.svg"
        self.memory_snapshot_dir = run_dir / "memory_snapshots"
        self.final_memory_path = run_dir / "final_memory.json"
        self.prediction_events_path = run_dir / "prediction_events.jsonl"
        self.training_rows: List[Dict[str, Any]] = []
        self.curve_rows: List[Dict[str, Any]] = []
        self.prediction_events_logging_disabled = False
        self.prediction_events_logging_error: str | None = None
        ensure_dir(self.run_dir)
        self._reset_outputs()

    def _reset_outputs(self) -> None:
        for path in [
            self.training_trace_jsonl_path,
            self.training_trace_csv_path,
            self.learning_curve_csv_path,
            self.learning_curve_svg_path,
            self.final_memory_path,
            self.prediction_events_path,
        ]:
            if path.exists():
                path.unlink()
        if self.memory_snapshot_dir.exists():
            shutil.rmtree(self.memory_snapshot_dir)
        ensure_dir(self.memory_snapshot_dir)

    def record_prompt_evaluation(self, row: Dict[str, Any]) -> None:
        normalized = {
            "iteration": int(row.get("iteration", 0)),
            "prompt_id": row.get("prompt_id"),
            "score": float(row.get("score", 0.0)),
            "accuracy": float(row.get("accuracy", 0.0)),
            "loss": float(row.get("loss", 0.0)),
            "misaligned_count": int(row.get("misaligned_count", 0)),
            "surrogate_metric": row.get("surrogate_metric", "heuristic_profile_alignment"),
            "evaluation_backend": row.get("evaluation_backend", "heuristic_profile_alignment"),
            "prompt_text": row.get("text", ""),
        }
        self.training_rows.append(normalized)
        append_jsonl(self.training_trace_jsonl_path, normalized)

    def record_iteration_summary(self, iteration: int, best_entry: Dict[str, Any]) -> None:
        row = {
            "iteration": int(iteration),
            "best_prompt_id": best_entry.get("prompt_id"),
            "best_score": float(best_entry.get("score", 0.0)),
            "best_accuracy": float(best_entry.get("accuracy", 0.0)),
            "best_loss": float(best_entry.get("loss", 0.0)),
            "best_misaligned_count": int(best_entry.get("misaligned_count", 0)),
        }
        self.curve_rows.append(row)
        write_csv(
            self.learning_curve_csv_path,
            self.curve_rows,
            ["iteration", "best_prompt_id", "best_score", "best_accuracy", "best_loss", "best_misaligned_count"],
        )
        self.learning_curve_svg_path.write_text(build_learning_curve_svg(self.curve_rows), encoding="utf-8")

    def write_memory_snapshot(self, iteration: int, entries: Sequence[Dict[str, Any]]) -> None:
        snapshot_path = self.memory_snapshot_dir / f"iteration_{int(iteration):03d}.json"
        write_json(
            snapshot_path,
            {
                "iteration": int(iteration),
                "top_prompts": [_snapshot_entry(entry) for entry in entries],
            },
        )

    def write_final_memory(self, entries: Sequence[Dict[str, Any]]) -> None:
        write_json(
            self.final_memory_path,
            {
                "method": self.method_name,
                "top_prompts": [_snapshot_entry(entry) for entry in entries],
            },
        )

    def record_prediction_event(self, event: Dict[str, Any]) -> None:
        if self.prediction_events_logging_disabled:
            return
        try:
            append_jsonl(self.prediction_events_path, event)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                self.prediction_events_logging_disabled = True
                self.prediction_events_logging_error = f"prediction_events_disabled:{type(exc).__name__}:{exc}"
                return
            raise

    def finalize(self) -> None:
        write_csv(
            self.training_trace_csv_path,
            self.training_rows,
            [
                "iteration",
                "prompt_id",
                "score",
                "accuracy",
                "loss",
                "misaligned_count",
                "surrogate_metric",
                "evaluation_backend",
                "prompt_text",
            ],
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "training_trace_jsonl": str(self.training_trace_jsonl_path),
            "training_trace_csv": str(self.training_trace_csv_path),
            "learning_curve_csv": str(self.learning_curve_csv_path),
            "learning_curve_svg": str(self.learning_curve_svg_path),
            "memory_snapshot_dir": str(self.memory_snapshot_dir),
            "final_memory": str(self.final_memory_path),
            "prediction_events_jsonl": str(self.prediction_events_path),
            "prediction_events_logging_disabled": self.prediction_events_logging_disabled,
            "prediction_events_logging_error": self.prediction_events_logging_error,
        }
