from __future__ import annotations

from typing import Any, Dict

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod, predict_from_profile


class VanillaMethod(BaseMethod):
    name = "vanilla"

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        pred: Any = predict_from_profile(self.task, sample)
        return {
            "prediction": pred,
            "selected_prompt_id": "vanilla_init",
            "rop_neighbor_ids": [],
        }

