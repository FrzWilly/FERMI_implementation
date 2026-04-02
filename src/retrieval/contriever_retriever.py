"""Dense retriever using facebook/contriever (mean-pooled transformer).

Load order:
  1. facebook/contriever via transformers (paper-faithful)
  2. sentence-transformers/all-mpnet-base-v2 via SentenceTransformer (fallback)
  3. BM25 lexical (fallback when no GPU/model available)

The effective backend is stored in self.backend for observability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from FERMI.src.retrieval.bm25_retriever import BM25LikeRetriever

_CONTRIEVER_MODEL = "facebook/contriever"
_ST_FALLBACK_MODEL = "sentence-transformers/all-mpnet-base-v2"


class ContrieverRetriever:
    def __init__(self, model_name: str = _CONTRIEVER_MODEL) -> None:
        self.model_name = model_name
        self.backend: str = "uninitialized"
        self._tokenizer: Any = None
        self._hf_model: Any = None
        self._st_encoder: Any = None
        self._bm25_fallback: Optional[BM25LikeRetriever] = None

        self._init_encoder(model_name)

    # ------------------------------------------------------------------
    # Initialisation — tries Contriever → ST → BM25
    # ------------------------------------------------------------------

    def _init_encoder(self, model_name: str) -> None:
        # 1) Contriever via HuggingFace transformers (mean-pool)
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = AutoModel.from_pretrained(model_name)
            self._hf_model.eval()
            self.backend = "contriever"
            return
        except Exception:
            pass

        # 2) sentence-transformers fallback
        try:
            from sentence_transformers import SentenceTransformer
            self._st_encoder = SentenceTransformer(_ST_FALLBACK_MODEL)
            self.backend = f"st:{_ST_FALLBACK_MODEL}"
            return
        except Exception:
            pass

        # 3) BM25 lexical fallback
        self._bm25_fallback = BM25LikeRetriever()
        self.backend = "bm25_fallback"

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "contriever":
            return self._embed_contriever(texts)
        if self.backend.startswith("st:") and self._st_encoder is not None:
            result: Any = self._st_encoder.encode(texts, show_progress_bar=False)
            return result.tolist()  # type: ignore[no-any-return]
        return []

    def _embed_contriever(self, texts: List[str]) -> List[List[float]]:
        import torch  # type: ignore[import-untyped]
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self._hf_model(**inputs)
        # Mean pooling over token dimension.
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = (token_embeddings * mask_expanded).sum(1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        embeddings = (sum_embeddings / sum_mask).cpu().tolist()
        return embeddings

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        candidates: Sequence[Dict],
        text_fn,
        top_k: int = 3,
    ) -> List[Dict]:
        if not candidates:
            return []

        if self._bm25_fallback is not None:
            return self._bm25_fallback.retrieve(
                query=query, candidates=candidates, text_fn=text_fn, top_k=top_k
            )

        candidate_texts = [text_fn(c) for c in candidates]
        all_texts = [query] + candidate_texts
        try:
            all_embeddings = self._embed(all_texts)
        except Exception:
            # Unexpected runtime error — degrade to BM25.
            self._bm25_fallback = BM25LikeRetriever()
            self.backend = "bm25_fallback"
            return self._bm25_fallback.retrieve(
                query=query, candidates=candidates, text_fn=text_fn, top_k=top_k
            )

        query_emb = all_embeddings[0]
        doc_embs = all_embeddings[1:]

        def _dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def _norm(v: List[float]) -> float:
            import math
            return math.sqrt(sum(x * x for x in v)) or 1e-9

        q_norm = _norm(query_emb)
        scores = [
            _dot(query_emb, d) / (q_norm * _norm(d))
            for d in doc_embs
        ]

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
