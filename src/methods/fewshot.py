from __future__ import annotations

from typing import Any, Dict

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod
from FERMI.src.methods.inference_utils import (
    build_fewshot_inference_prompt,
    build_prediction_repair_prompt,
    parse_llm_prediction,
)
from FERMI.src.methods.llm_interface import LLMClient
from FERMI.src.retrieval.bm25_retriever import BM25LikeRetriever
from FERMI.src.retrieval.contriever_retriever import ContrieverRetriever


class FewShotMethod(BaseMethod):
    """Retrieve top-k profile items, build Listing-3 prompt, call LLM M.

    Matches the paper's Few-shot baseline: retrieved QA pairs are formatted
    into the prompt template and sent to LLM M for the final prediction.
    Falls back to rule-based averaging only when the LLM call fails.
    """

    name = "fewshot"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.top_k = int(self.config.get("retrieval_topk", 3))
        retriever_name = str(self.config.get("retriever", "bm25")).lower()
        self.retriever = BM25LikeRetriever() if retriever_name == "bm25" else ContrieverRetriever()

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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        retrieved = self.retriever.retrieve(
            query=sample.input_text,
            candidates=sample.profile,
            text_fn=self.profile_text_fn,
            top_k=self.top_k,
        )

        prompt = build_fewshot_inference_prompt(sample, retrieved)
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
            "selected_prompt_id": f"fewshot_top{self.top_k}",
            "rop_neighbor_ids": [str(x.get("id")) for x in retrieved[: self.top_k]],
            "prediction_source": "llm",
        }

    def runtime_summary(self) -> Dict[str, Any]:
        retriever_backend = getattr(self.retriever, "backend", type(self.retriever).__name__)
        return {
            "llm": self.llm.summary(),
            "model_M_name": self.model_m_name,
            "retriever": retriever_backend,
            "top_k": self.top_k,
            "prediction_source_counts": self._prediction_source_counts,
        }
