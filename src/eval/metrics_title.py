from __future__ import annotations

import re
from typing import Sequence


TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(pred: str, gold: str) -> float:
    p = _tokens(pred)
    g = _tokens(gold)
    if not p or not g:
        return 0.0
    lcs = _lcs_len(p, g)
    precision = lcs / len(p)
    recall = lcs / len(g)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def mean_rouge_l(preds: Sequence[str], golds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    scores = [rouge_l_f1(str(p), str(g)) for p, g in zip(preds, golds)]
    return sum(scores) / len(scores)

