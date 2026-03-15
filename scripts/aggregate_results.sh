#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import json
from pathlib import Path

root = Path("FERMI/results")
rows = []
for run_dir in sorted([p for p in root.glob("*") if p.is_dir()]):
    metric_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.json"
    if not metric_path.exists() or not summary_path.exists():
        continue
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "run_id": summary.get("run_id"),
            "task": summary.get("task"),
            "method": summary.get("method"),
            "split": summary.get("split"),
            "primary_metric": metrics.get("primary_metric"),
            "metric_value": metrics.get("metric_value"),
            "n_samples": metrics.get("n_samples"),
        }
    )

out_path = root / "leaderboard_rows.json"
out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(rows)} rows to {out_path}")
PY

