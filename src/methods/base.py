from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample, profile_item_text


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
        tags = [str(p.get("tag", "")) for p in profile if p.get("tag")]
        if not tags:
            return "classic"
        return Counter(tags).most_common(1)[0][0]

    if task == "LaMP3_rate":
        scores = [_safe_int(p.get("score", 3), default=3) for p in profile]
        avg = round(sum(scores) / max(len(scores), 1))
        return max(1, min(5, avg))

    if task == "LaMP5_title":
        # 以最接近 query 的 profile title 作為 fallback（此處先取最長 title）。
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

    def fit(self, train_samples: List[UnifiedSample]) -> None:
        self.train_samples = train_samples

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        raise NotImplementedError

    def profile_text_fn(self, item: Dict) -> str:
        return profile_item_text(self.task, item)

    def runtime_summary(self) -> Dict[str, Any]:
        """供 runner 寫入 summary.json 的方法執行摘要。"""
        return {}
