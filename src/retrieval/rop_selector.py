from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

from FERMI.src.retrieval.bm25_retriever import lexical_overlap_score


class RoPSelector:
    def __init__(
        self,
        n_tilde: int = 3,
        backend: str = "mpnet",
        mpnet_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        self.n_tilde = n_tilde
        self.backend_requested = str(backend).lower()
        self.mpnet_model_name = mpnet_model_name

        self.backend_effective = self.backend_requested
        self.fallback_used = False
        self.fallback_reason = ""
        self._mpnet_model = None
        self._embedding_cache: Dict[str, List[float]] = {}
        self._mpnet_device = None

        if self.backend_requested == "mpnet":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._mpnet_model = SentenceTransformer(self.mpnet_model_name)
                self._mpnet_device = str(getattr(self._mpnet_model, "device", "unknown"))
                self.backend_effective = "mpnet"
            except Exception as exc:  # pragma: no cover - runtime dependent
                self._activate_fallback(f"mpnet_unavailable:{type(exc).__name__}:{exc}")
        elif self.backend_requested in {"lexical", "bm25"}:
            self.backend_effective = "lexical"
        else:
            self._activate_fallback(f"unsupported_backend:{self.backend_requested}")

    def _activate_fallback(self, reason: str) -> None:
        self.backend_effective = "lexical"
        self.fallback_used = True
        if not self.fallback_reason:
            self.fallback_reason = reason

    def _encode_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if self.backend_effective != "mpnet" or self._mpnet_model is None:
            return []

        normalized = [str(text) for text in texts]
        missing = [text for text in normalized if text not in self._embedding_cache]

        if missing:
            try:
                embeddings = self._mpnet_model.encode(
                    missing,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
                if hasattr(embeddings, "detach"):
                    embeddings = embeddings.detach().cpu()
                    rows = embeddings.tolist()
                    if missing and rows and isinstance(rows[0], (int, float)):
                        rows = [rows]
                else:
                    rows = []
                    for emb in embeddings:
                        if hasattr(emb, "detach"):
                            rows.append(emb.detach().cpu().tolist())
                        elif hasattr(emb, "tolist"):
                            rows.append(emb.tolist())
                        else:
                            rows.append(list(emb))

                for text, vector in zip(missing, rows):
                    self._embedding_cache[text] = [float(v) for v in vector]
            except Exception as exc:  # pragma: no cover - runtime dependent
                self._activate_fallback(f"mpnet_runtime_error:{type(exc).__name__}:{exc}")
                return []

        vectors: List[List[float]] = []
        for text in normalized:
            vector = self._embedding_cache.get(text)
            if vector is None:
                self._activate_fallback("mpnet_runtime_error:MissingEmbeddingCache")
                return []
            vectors.append(vector)
        return vectors

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sum(float(a[i]) * float(b[i]) for i in range(n))
        na = math.sqrt(sum(float(a[i]) * float(a[i]) for i in range(n)))
        nb = math.sqrt(sum(float(b[i]) * float(b[i]) for i in range(n)))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    def _query_text_similarity(self, query: str, text: str) -> float:
        if self.backend_effective == "mpnet" and self._mpnet_model is not None:
            vecs = self._encode_texts([query, text])
            if len(vecs) == 2:
                qv, tv = vecs
                return self._cosine(qv, tv)

        return lexical_overlap_score(query, text)

    def _top_relevant_opinion_ids(self, query: str, opinion_samples: Sequence[Dict[str, Any]]) -> List[str]:
        if not opinion_samples:
            return []
        scored: List[Tuple[float, str]] = []
        for s in opinion_samples:
            sid = str(s.get("id", ""))
            q_text = str(s.get("input_text", ""))
            if not sid:
                continue
            sim = self._query_text_similarity(query, q_text)
            scored.append((sim, sid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sid for _, sid in scored[: self.n_tilde]]

    def select(
        self,
        query: str,
        prompt_pool: Sequence[Dict],
        opinion_samples: Sequence[Dict[str, Any]] | None = None,
    ) -> Tuple[Dict, List[str]]:
        if not prompt_pool:
            return {"prompt_id": "none", "text": ""}, []

        relevant_ids = self._top_relevant_opinion_ids(query, opinion_samples or []) if opinion_samples else []

        scored = []
        for p in prompt_pool:
            score = 0.0
            sample_scores = p.get("sample_scores", {})
            if relevant_ids and isinstance(sample_scores, dict):
                matched = [rid for rid in relevant_ids if rid in sample_scores]
                if matched:
                    vals = [float(sample_scores.get(rid, 0.0)) for rid in matched]
                    score = sum(vals) / len(vals) if vals else 0.0
                else:
                    score = self._query_text_similarity(query, str(p.get("text", "")))
            else:
                score = self._query_text_similarity(query, str(p.get("text", "")))
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)

        neighbors = [str(p.get("prompt_id")) for _, p in scored[: self.n_tilde]]
        selected = scored[0][1]
        return selected, neighbors

    def summary(self) -> Dict[str, Any]:
        return {
            "n_tilde": self.n_tilde,
            "backend_requested": self.backend_requested,
            "backend_effective": self.backend_effective,
            "mpnet_model_name": self.mpnet_model_name,
            "mpnet_device": self._mpnet_device,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason or None,
        }
