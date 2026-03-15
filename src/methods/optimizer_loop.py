from __future__ import annotations

from typing import Dict, List, Sequence

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.eval.metrics_title import rouge_l_f1
from FERMI.src.methods.base import predict_from_profile


def _sample_score(task: str, pred, gold) -> float:
    if gold is None:
        return 0.0
    if task == "LaMP2_tag":
        return 1.0 if str(pred) == str(gold) else 0.0
    if task == "LaMP3_rate":
        # mae 越低越好，轉成越高越好的分數
        return max(0.0, 1.0 - abs(float(pred) - float(gold)) / 4.0)
    if task == "LaMP5_title":
        return rouge_l_f1(str(pred), str(gold))
    return 0.0


def score_prompt(task: str, prompt_text: str, samples: Sequence[UnifiedSample]) -> float:
    if not samples:
        return 0.0
    # Phase 1: prompt_text 不影響規則式預測器，但保留介面。
    scores: List[float] = []
    for s in samples:
        pred = predict_from_profile(task, s)
        scores.append(_sample_score(task, pred, s.gold))
    return sum(scores) / len(scores)


def collect_misaligned_context(task: str, samples: Sequence[UnifiedSample], max_items: int = 8) -> List[str]:
    misaligned: List[str] = []
    for s in samples:
        if s.gold is None:
            continue
        pred = predict_from_profile(task, s)
        if str(pred) != str(s.gold):
            misaligned.append(f"id={s.id} pred={pred} gold={s.gold} input={s.input_text[:180]}")
        if len(misaligned) >= max_items:
            break
    return misaligned


def generate_k_prompts(
    seed_prompt: str,
    misaligned_context: Sequence[str],
    k: int,
    generator_fn,
    temperature: float = 1.0,
) -> List[str]:
    context = "\n".join(misaligned_context[:5])
    prompt = (
        "Generate improved instruction prompts for personalization tasks.\n"
        f"Current prompt:\n{seed_prompt}\n\n"
        f"Mis-aligned context examples:\n{context}\n\n"
        f"Return one improved prompt."
    )
    new_prompts: List[str] = []
    for _ in range(k):
        out = generator_fn(prompt, temperature)
        out = out.strip() if out else ""
        if not out:
            out = seed_prompt + " | refine with clearer constraints"
        new_prompts.append(out)
    return new_prompts

