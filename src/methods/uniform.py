from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod


class UniformMethod(BaseMethod):
    name = "uniform"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.global_prior: Any = None

    def fit(self, train_samples: list[UnifiedSample]) -> None:
        super().fit(train_samples)
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
        return {
            "prediction": self.global_prior,
            "selected_prompt_id": None,
            "rop_neighbor_ids": [],
        }

