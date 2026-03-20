from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample, profile_item_text


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set:
    return set(TOKEN_RE.findall(str(text).lower()))


def _best_match_by_overlap(query: str, candidates: List[Dict], text_field: str) -> Dict | None:
    qtok = _tokenize(query)
    best = None
    best_score = -1.0
    for item in candidates:
        text = str(item.get(text_field, ""))
        ttok = _tokenize(text)
        score = 0.0
        if qtok and ttok:
            score = len(qtok.intersection(ttok)) / max(len(qtok), 1)
        if score > best_score:
            best = item
            best_score = score
    return best


def _safe_int(value: Any, default: int = 3) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def predict_from_profile(task: str, sample: UnifiedSample, profile_subset: List[Dict] | None = None) -> Any:
    profile = profile_subset if profile_subset is not None else sample.profile
    if not profile:
        if task == "LaMP2_tag":
            return "classic"
        if task == "LaMP3_rate":
            return 3
        return "Untitled"

    if task == "LaMP2_tag":
        best = _best_match_by_overlap(sample.input_text, profile, "description")
        if best and best.get("tag"):
            return str(best.get("tag"))
        tags = [str(p.get("tag", "")) for p in profile if p.get("tag")]
        if not tags:
            return "classic"
        return Counter(tags).most_common(1)[0][0]

    if task == "LaMP3_rate":
        best = _best_match_by_overlap(sample.input_text, profile, "text")
        if best and best.get("score") is not None:
            return max(1, min(5, _safe_int(best.get("score"), default=3)))
        scores = [_safe_int(p.get("score", 3), default=3) for p in profile]
        avg = round(sum(scores) / max(len(scores), 1))
        return max(1, min(5, avg))

    if task == "LaMP5_title":
        best = _best_match_by_overlap(sample.input_text, profile, "abstract")
        if best and best.get("title"):
            return str(best.get("title")).strip()

        # fallback：取最長 title。
        titles = [str(p.get("title", "")).strip() for p in profile if p.get("title")]
        if not titles:
            return "Untitled"
        return sorted(titles, key=len, reverse=True)[0]

    return ""


class BaseMethod:
    name = "base"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        self.task = task
        self.config = config or {}
        self.train_samples: List[UnifiedSample] = []
        self.run_dir: Path | None = None
        self.run_split: str | None = None

    def fit(self, train_samples: List[UnifiedSample]) -> None:
        self.train_samples = train_samples

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        raise NotImplementedError

    def set_run_context(self, run_dir: Path, split: str) -> None:
        self.run_dir = run_dir
        self.run_split = split

    def profile_text_fn(self, item: Dict) -> str:
        return profile_item_text(self.task, item)

    def artifact_summary(self) -> Dict[str, Any]:
        return {}

    def runtime_summary(self) -> Dict[str, Any]:
        """供 runner 寫入 summary.json 的方法執行摘要。"""
        return {}
