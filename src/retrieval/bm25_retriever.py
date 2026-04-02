"""BM25 retriever with proper IDF/TF weighting (Okapi BM25).

k1 = 1.5, b = 0.75 — standard defaults from the original BM25 paper.
IDF is computed over the provided candidates corpus (the user's profile items).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def _tf_counter(tokens: List[str]) -> Dict[str, int]:
    return dict(Counter(tokens))


def lexical_overlap_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    text_tokens = set(_tokenize(text))
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = len(query_tokens.intersection(text_tokens))
    denom = math.sqrt(len(query_tokens) * len(text_tokens))
    return overlap / denom if denom else 0.0


def bm25_scores(
    query_tokens: List[str],
    docs_tokens: List[List[str]],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[float]:
    """Return BM25 score for each document against the query."""
    N = len(docs_tokens)
    if N == 0 or not query_tokens:
        return [0.0] * N

    # Document lengths and average length.
    doc_lens = [len(d) for d in docs_tokens]
    avgdl = sum(doc_lens) / max(N, 1)

    # Document frequency per query term (over this mini-corpus).
    df: Dict[str, int] = {}
    for term in set(query_tokens):
        df[term] = sum(1 for d in docs_tokens if term in d)

    # IDF per query term (Robertson-Sparck Jones with smoothing).
    idf: Dict[str, float] = {
        term: math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        for term in set(query_tokens)
    }

    scores: List[float] = []
    for idx, doc in enumerate(docs_tokens):
        tf_map = _tf_counter(doc)
        dl = doc_lens[idx]
        score = 0.0
        for term in set(query_tokens):
            tf = tf_map.get(term, 0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf.get(term, 0.0) * numerator / max(denominator, 1e-9)
        scores.append(score)

    return scores


class BM25LikeRetriever:
    """Okapi BM25 retriever computed over the query-time candidate corpus."""

    def retrieve(
        self,
        query: str,
        candidates: Sequence[Dict],
        text_fn,
        top_k: int = 3,
    ) -> List[Dict]:
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        docs_tokens = [_tokenize(text_fn(c)) for c in candidates]

        if not query_tokens:
            return list(candidates[:top_k])

        scores = bm25_scores(query_tokens, docs_tokens)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
