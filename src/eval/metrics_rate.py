from __future__ import annotations

from typing import Sequence


def mae(preds: Sequence[float], golds: Sequence[float]) -> float:
    if not golds:
        return 0.0
    err = 0.0
    for p, g in zip(preds, golds):
        err += abs(float(p) - float(g))
    return err / len(golds)

