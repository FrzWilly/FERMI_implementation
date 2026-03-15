from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from FERMI.src.retrieval.bm25_retriever import lexical_overlap_score


class RoPSelector:
    def __init__(self, n_tilde: int = 3) -> None:
        self.n_tilde = n_tilde

    def select(self, query: str, prompt_pool: Sequence[Dict]) -> Tuple[Dict, List[str]]:
        if not prompt_pool:
            return {"prompt_id": "none", "text": ""}, []

        scored = []
        for p in prompt_pool:
            score = lexical_overlap_score(query, str(p.get("text", "")))
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        neighbors = [str(p.get("prompt_id")) for _, p in scored[: self.n_tilde]]
        selected = scored[0][1]
        return selected, neighbors

