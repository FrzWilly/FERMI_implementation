from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Dict

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod


class UniformMethod(BaseMethod):
    name = "uniform"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.global_prior: Any = None
        self.seed = int(self.config.get("seed", 42))

    def _uniform_rate_prediction(self, sample: UnifiedSample) -> int:
        sample_key = f"{self.seed}:{sample.user_id}:{sample.id}"
        digest = hashlib.sha256(sample_key.encode("utf-8")).hexdigest()
        return (int(digest[:8], 16) % 5) + 1

    def fit(self, train_samples: list[UnifiedSample]) -> None:
        super().fit(train_samples)
        if self.task == "LaMP3_rate":
            # Paper baseline uses uniformly random predictions instead of a
            # train-set prior. We keep the prediction deterministic per sample
            # for reproducible reruns under a fixed seed.
            self.global_prior = None
            return

        golds = [s.gold for s in train_samples if s.gold is not None]
        if not golds:
            self.global_prior = "classic" if self.task == "LaMP2_tag" else 3
            return

        if self.task == "LaMP2_tag":
            self.global_prior = Counter([str(g) for g in golds]).most_common(1)[0][0]
        elif self.task == "LaMP3_rate":
            self.global_prior = round(sum(int(g) for g in golds) / len(golds))
        elif self.task == "LaMP5_title":
            self.global_prior = Counter([str(g) for g in golds]).most_common(1)[0][0]

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        prediction = self._uniform_rate_prediction(sample) if self.task == "LaMP3_rate" else self.global_prior
        return {
            "prediction": prediction,
            "selected_prompt_id": "uniform_random" if self.task == "LaMP3_rate" else None,
            "rop_neighbor_ids": [],
        }
