from __future__ import annotations

import re
from typing import Dict, List, Sequence


TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


def lexical_overlap_score(query: str, text: str) -> float:
    q = _tokenize(query)
    t = _tokenize(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    union = len(q | t)
    return inter / max(union, 1)


class BM25LikeRetriever:
    """以 lexical overlap 作為 BM25-like fallback。"""

    def retrieve(self, query: str, candidates: Sequence[Dict], text_fn, top_k: int = 3) -> List[Dict]:
        scored = []
        for c in candidates:
            score = lexical_overlap_score(query, text_fn(c))
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

