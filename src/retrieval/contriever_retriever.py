from __future__ import annotations

from typing import Dict, List, Sequence

from FERMI.src.retrieval.bm25_retriever import BM25LikeRetriever


class ContrieverRetriever:
    """
    dense retriever 介面 placeholder。
    Phase 1 預設 fallback 到 lexical overlap。
    """

    def __init__(self) -> None:
        self._fallback = BM25LikeRetriever()

    def retrieve(self, query: str, candidates: Sequence[Dict], text_fn, top_k: int = 3) -> List[Dict]:
        return self._fallback.retrieve(query=query, candidates=candidates, text_fn=text_fn, top_k=top_k)

