from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config_file(path: Path) -> dict:
    """
    先嘗試 JSON（JSON 是 YAML 子集）；若失敗再嘗試 pyyaml（若環境可用）。
    """
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Config root must be object: {path}")
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"Config root must be mapping: {path}")
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ValueError(f"Unable to parse config file: {path}") from exc

