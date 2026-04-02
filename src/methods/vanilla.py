from __future__ import annotations

from typing import Any, Dict

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod
from FERMI.src.methods.inference_utils import (
    build_vanilla_inference_prompt,
    build_prediction_repair_prompt,
    parse_llm_prediction,
)
from FERMI.src.methods.llm_interface import LLMClient


class VanillaMethod(BaseMethod):
    """Call LLM M with only the task instruction and query — no user history.

    Matches the paper's Vanilla baseline: no personalization context is passed.
    Falls back to the rule-based heuristic only when the LLM call fails.
    """

    name = "vanilla"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.model_m_name = str(self.config.get("model_M_name", "gpt-3.5-turbo"))
        self.m_temperature = float(self.config.get("M_temperature", self.config.get("model_M_temperature", 0.0)))
        self.prediction_max_tokens = int(self.config.get("prediction_max_tokens", 64))

        self.llm = LLMClient(
            evaluator_model=self.model_m_name,
            api_key_env=str(self.config.get("openai_api_key_env", "OPENAI_API_KEY")),
            max_retries=int(self.config.get("openai_max_retries", 2)),
            request_timeout=float(self.config.get("openai_request_timeout", 30.0)),
        )
        self._prediction_source_counts: Dict[str, int] = {"llm": 0}

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        prompt = build_vanilla_inference_prompt(self.task, sample)
        raw = self.llm.generate(
            prompt,
            temperature=self.m_temperature,
            max_tokens=self.prediction_max_tokens,
            model=self.model_m_name,
            role="evaluator",
            allow_fallback=False,
        )
        pred = parse_llm_prediction(self.task, raw)
        if pred is None:
            repair_prompt = build_prediction_repair_prompt(self.task, raw)
            repaired_raw = self.llm.generate(
                repair_prompt,
                temperature=0.0,
                max_tokens=16,
                model=self.model_m_name,
                role="evaluator",
                allow_fallback=False,
            )
            pred = parse_llm_prediction(self.task, repaired_raw)
            if pred is None:
                raise RuntimeError(
                    f"LLM parse failed for sample {sample.id}: raw={raw!r}, repaired_raw={repaired_raw!r}"
                )
        self._prediction_source_counts["llm"] += 1
        return {
            "prediction": pred,
            "selected_prompt_id": "vanilla_init",
            "rop_neighbor_ids": [],
            "prediction_source": "llm",
        }

    def runtime_summary(self) -> Dict[str, Any]:
        return {
            "llm": self.llm.summary(),
            "model_M_name": self.model_m_name,
            "prediction_source_counts": self._prediction_source_counts,
        }
