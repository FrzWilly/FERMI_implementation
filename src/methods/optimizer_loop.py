from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.eval.metrics_title import rouge_l_f1
from FERMI.src.methods.base import predict_from_profile
from FERMI.src.prompts.templates import FERMI_FIGURE8_POPT, OPRO_FIGURE9_POPT


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


def evaluate_prompt_on_samples(
    task: str,
    prompt_text: str,
    samples: Sequence[UnifiedSample],
    tau: float = 1.0,
    predictor_fn: Callable[[UnifiedSample], Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    if not samples:
        return {
            "score": 0.0,
            "sample_scores": {},
            "misaligned_records": [],
            "misaligned_indices": set(),
        }

    scores: List[float] = []
    sample_scores: Dict[str, float] = {}
    misaligned_records: List[Dict[str, Any]] = []
    misaligned_indices = set()
    predictions: Dict[str, Any] = {}
    prediction_sources: Dict[str, int] = {}

    threshold = float(tau)
    exact_matches = 0
    for s in samples:
        if predictor_fn is None:
            pred = predict_from_profile(task, s)
            source = "heuristic_profile_alignment"
        else:
            pred_result = predictor_fn(s)
            pred = pred_result.get("prediction")
            source = str(pred_result.get("prediction_source", "llm_prompt_inference"))
        score = _sample_score(task, pred, s.gold)
        scores.append(score)
        sample_scores[str(s.id)] = score
        predictions[str(s.id)] = pred
        prediction_sources[source] = prediction_sources.get(source, 0) + 1
        if s.gold is not None and str(pred) == str(s.gold):
            exact_matches += 1

    for idx, s in enumerate(samples, start=1):
        pred = predictions.get(str(s.id))
        score = sample_scores.get(str(s.id), 0.0)
        if score < threshold:
            misaligned_indices.add(idx)
            misaligned_records.append(
                {
                    "index": idx,
                    "id": str(s.id),
                    "question": str(s.input_text),
                    "gold": s.gold,
                    "prediction": pred,
                    "score": score,
                }
            )

    mean_score = sum(scores) / len(scores)
    source_parts = [f"{name}:{count}" for name, count in sorted(prediction_sources.items())]

    return {
        "score": mean_score,
        "accuracy": exact_matches / len(samples),
        "loss": 1.0 - mean_score,
        "sample_scores": sample_scores,
        "misaligned_records": misaligned_records,
        "misaligned_indices": misaligned_indices,
        "misaligned_count": len(misaligned_records),
        "surrogate_metric": "llm_prompt_inference" if predictor_fn is not None else "heuristic_profile_alignment",
        "evaluation_backend": " | ".join(source_parts) if source_parts else "heuristic_profile_alignment",
    }


def score_prompt(task: str, prompt_text: str, samples: Sequence[UnifiedSample], tau: float = 1.0) -> float:
    result = evaluate_prompt_on_samples(task=task, prompt_text=prompt_text, samples=samples, tau=tau)
    return float(result.get("score", 0.0))


def collect_misaligned_context(task: str, samples: Sequence[UnifiedSample], max_items: int = 8) -> List[str]:
    result = evaluate_prompt_on_samples(task=task, prompt_text="", samples=samples, tau=1.0)
    records = list(result.get("misaligned_records", []))[:max_items]
    return [
        f"<{r.get('index')}> Question: {str(r.get('question', ''))[:180]} | "
        f"Answer: {r.get('gold')} | Your response: {r.get('prediction')}"
        for r in records
    ]


def build_demonstration_block(samples: Sequence[UnifiedSample], max_items: int = 4) -> str:
    demos: List[str] = []
    picked = list(samples)[: max(1, max_items)]
    if not picked:
        return "[1]\n<INS>\nQuestion: N/A\nAnswer: N/A"

    for idx, s in enumerate(picked, start=1):
        demos.append(f"[{idx}]")
        demos.append("<INS>")
        demos.append(f"Question: {s.input_text}")
        demos.append(f"Answer: {s.gold}")
    return "\n".join(demos)


def build_opro_memory_block(memory_entries_asc: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, e in enumerate(memory_entries_asc, start=1):
        lines.append(f"text: Prompt #{idx}")
        lines.append(str(e.get("text", "")))
        lines.append(f"score: {float(e.get('score', 0.0)):.4f}")
    return "\n".join(lines)


def _format_failure_case_line(record: Dict[str, Any]) -> str:
    return (
        f"<{record.get('index')}>\n"
        f"Question: {record.get('question', '')}\n"
        f"Answer: {record.get('gold')}\n"
        f"Your response: {record.get('prediction')}"
    )


def build_fermi_memory_block(memory_entries_asc: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    top1_indices: set = set()

    for idx, e in enumerate(memory_entries_asc, start=1):
        records = list(e.get("misaligned_records", []))
        indices = set(e.get("misaligned_indices", set()))

        lines.append(f"text: Prompt #{idx}")
        lines.append(str(e.get("text", "")))
        lines.append(f"score: {float(e.get('score', 0.0)):.4f}")
        lines.append("failure cases:")

        if idx == 1:
            top1_indices = set(indices)
            if not records:
                lines.append("None")
            else:
                for r in records[:8]:
                    lines.append(_format_failure_case_line(r))
        else:
            common = sorted(top1_indices.intersection(indices))
            newly_misaligned = max(0, len(indices - top1_indices))
            lines.append(f"{common} and {newly_misaligned} additional newly mis-aligned examples")

    return "\n".join(lines)


def build_opro_popt_prompt(memory_entries_asc: Sequence[Dict[str, Any]], demo_samples: Sequence[UnifiedSample]) -> str:
    memory_block = build_opro_memory_block(memory_entries_asc)
    demonstration_block = build_demonstration_block(demo_samples)
    return OPRO_FIGURE9_POPT.format(
        memory_block=memory_block,
        demonstration_block=demonstration_block,
    )


def build_fermi_popt_prompt(
    memory_entries_asc: Sequence[Dict[str, Any]],
    demo_samples: Sequence[UnifiedSample],
    task_focus: str,
    num_questions: int,
) -> str:
    memory_block = build_fermi_memory_block(memory_entries_asc)
    demonstration_block = build_demonstration_block(demo_samples)
    return FERMI_FIGURE8_POPT.format(
        task_focus=task_focus,
        num_questions=num_questions,
        memory_block=memory_block,
        demonstration_block=demonstration_block,
    )


def generate_k_prompts(
    prompt_for_mopt: str,
    k: int,
    generator_fn,
    temperature: float = 1.0,
) -> List[str]:
    from FERMI.src.methods.inference_utils import extract_optimized_prompt_candidate

    new_prompts: List[str] = []
    for _ in range(k):
        out = generator_fn(prompt_for_mopt, temperature)
        out = out.strip() if out else ""
        if not out:
            out = "[Refined prompt: preserve user preference alignment and task constraints.]"
        new_prompts.append(extract_optimized_prompt_candidate(out))
    return new_prompts
