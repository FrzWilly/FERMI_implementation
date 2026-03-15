from __future__ import annotations

from typing import Sequence


def accuracy(preds: Sequence[str], golds: Sequence[str]) -> float:
    if not golds:
        return 0.0
    correct = sum(1 for p, g in zip(preds, golds) if str(p) == str(g))
    return correct / len(golds)

