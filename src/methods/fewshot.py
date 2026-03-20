from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.methods.base import BaseMethod, predict_from_profile
from FERMI.src.prompts.formatter import build_fewshot_listing3_prompt
from FERMI.src.retrieval.bm25_retriever import BM25LikeRetriever
from FERMI.src.retrieval.contriever_retriever import ContrieverRetriever


class FewShotMethod(BaseMethod):
    name = "fewshot"

    def __init__(self, task: str, config: Dict | None = None) -> None:
        super().__init__(task, config)
        self.top_k = int(self.config.get("retrieval_topk", 3))
        retriever_name = str(self.config.get("retriever", "bm25")).lower()
        self.retriever = BM25LikeRetriever() if retriever_name == "bm25" else ContrieverRetriever()

    def _predict_with_retrieved(self, sample: UnifiedSample, retrieved: List[Dict]) -> Any:
        if not retrieved:
            return predict_from_profile(self.task, sample)

        if self.task == "LaMP2_tag":
            tags = [str(r.get("tag", "")) for r in retrieved if r.get("tag")]
            return Counter(tags).most_common(1)[0][0] if tags else "classic"

        if self.task == "LaMP3_rate":
            scores = [int(float(r.get("score", 3))) for r in retrieved]
            return max(1, min(5, round(sum(scores) / max(len(scores), 1))))

        if self.task == "LaMP5_title":
            titles = [str(r.get("title", "")).strip() for r in retrieved if r.get("title")]
            return titles[0] if titles else "Untitled"

        return ""

    def predict(self, sample: UnifiedSample) -> Dict[str, Any]:
        retrieved = self.retriever.retrieve(
            query=sample.input_text,
            candidates=sample.profile,
            text_fn=self.profile_text_fn,
            top_k=self.top_k,
        )
        _ = build_fewshot_listing3_prompt(sample, retrieved)
        pred = self._predict_with_retrieved(sample, retrieved)
        return {
            "prediction": pred,
            "selected_prompt_id": f"fewshot_top{self.top_k}",
            "rop_neighbor_ids": [str(x.get("id")) for x in retrieved[: self.top_k]],
        }
