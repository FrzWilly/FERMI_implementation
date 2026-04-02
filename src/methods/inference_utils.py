from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from FERMI.src.data.lamp_parser import UnifiedSample


_RATE_WORD_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def _clip_text(text: str, max_chars: int = 420) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= max_chars:
        return clean
    return f"{clean[: max_chars - 3]}..."


def build_profile_history_block(task: str, profile: List[Dict[str, Any]], max_items: int = 20) -> str:
    if not profile:
        return "(empty profile)"

    lines: List[str] = []
    for idx, item in enumerate(profile[: max(1, max_items)], start=1):
        lines.append(f"[{idx}] id={item.get('id', '')}")
        if task == "LaMP2_tag":
            lines.append(f"Description: {_clip_text(str(item.get('description', '')))}")
            lines.append(f"Label: {str(item.get('tag', ''))}")
        elif task == "LaMP3_rate":
            lines.append(f"Review: {_clip_text(str(item.get('text', '')))}")
            lines.append(f"Rating: {str(item.get('score', ''))}")
        else:
            lines.append(f"Title: {_clip_text(str(item.get('title', '')))}")
            lines.append(f"Abstract: {_clip_text(str(item.get('abstract', '')))}")
        lines.append("")
    return "\n".join(lines).strip()


def _task_output_instruction(task: str) -> str:
    if task == "LaMP2_tag":
        return "Return exactly one tag label and nothing else."
    if task == "LaMP3_rate":
        return "Return exactly one integer from 1 to 5 and nothing else."
    if task == "LaMP5_title":
        return "Return exactly one concise title and nothing else."
    return "Return only the final answer."


_TAG_ANSWER_CHOICES = (
    "A. sci-fi B. based on a book C. comedy D. action E. twist ending "
    "F. dystopia G. dark comedy H. classic I. psychology J. fantasy K. romance "
    "L. thought-provoking M. social commentary N. violence O. true story"
)


def build_vanilla_inference_prompt(task: str, sample: UnifiedSample) -> str:
    """Build a vanilla prompt that contains NO user-history — just the task question.

    Matches the paper's Vanilla baseline: M is called with only the task instruction
    and the test query, without any profile context.
    """
    from FERMI.src.prompts.templates import DEFAULT_TEMPLATES

    template = DEFAULT_TEMPLATES.get(task, "{question}\nAnswer:")
    answer_choices = _TAG_ANSWER_CHOICES if task == "LaMP2_tag" else "N/A"
    return template.format(question=sample.input_text, answer_choices=answer_choices)


def build_fewshot_inference_prompt(sample: UnifiedSample, retrieved: List[Dict[str, Any]]) -> str:
    """Build the few-shot prompt (Listing 3) and return it ready to send to LLM M.

    Delegates to the existing build_fewshot_listing3_prompt() in formatter.py;
    task is inferred from sample.task internally.
    """
    from FERMI.src.prompts.formatter import build_fewshot_listing3_prompt

    return build_fewshot_listing3_prompt(sample, retrieved)


def build_personalized_inference_prompt(
    task: str,
    optimized_prompt: str,
    sample: UnifiedSample,
) -> str:
    """Paper-aligned inference: M(q; p'_k*) per Algorithm 1.

    The optimized prompt replaces <INS>; the test question follows.
    User history is NOT included — it was only used during optimization (Mopt),
    not during M's prediction at inference time.

    Format matches Figure 17 (LaMP3_rate) / Figure 18 (LaMP2_tag, LaMP5_title).
    """
    prompt = optimized_prompt.strip()
    if task == "LaMP3_rate":
        # Figure 17 format
        return (
            f"{prompt} Just answer with 1, 2, 3, 4, or 5 without further explanation:\n\n"
            f"Question: {sample.input_text}\n\n"
            f"Answer:"
        )
    if task == "LaMP2_tag":
        # Figure 18 format (tag classification)
        return (
            f"{prompt}\n"
            f"Question: {sample.input_text}\n"
            f"Answer choices: {_TAG_ANSWER_CHOICES}\n"
            f"Answer:"
        )
    # LaMP5_title and others — Figure 18 style
    return (
        f"{prompt}\n"
        f"Question: {sample.input_text}\n"
        f"Answer:"
    )


def extract_optimized_prompt_candidate(raw_text: str) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""

    bracket_matches = re.findall(r"\[(.*?)\]", text, flags=re.DOTALL)
    if bracket_matches:
        candidate = bracket_matches[-1].strip()
        if candidate:
            return candidate

    first_block = text.split("\n\n", 1)[0].strip()
    return first_block or text


def parse_llm_prediction(task: str, raw_text: str) -> Optional[Any]:
    text = str(raw_text or "").strip()
    if not text:
        return None

    if task == "LaMP3_rate":
        digit_match = re.search(r"\b([1-5])\b", text)
        if digit_match:
            return int(digit_match.group(1))
        any_int_match = re.search(r"(?<!\d)(-?\d+)(?!\d)", text)
        if any_int_match:
            value = int(any_int_match.group(1))
            return max(1, min(5, value))
        lowered = text.lower()
        for word, value in _RATE_WORD_MAP.items():
            if re.search(rf"\b{word}\b", lowered):
                return value
        return None

    if task == "LaMP2_tag":
        first_line = text.splitlines()[0].strip()
        first_line = re.sub(r"^(answer|label)\s*:\s*", "", first_line, flags=re.IGNORECASE)
        return first_line or None

    if task == "LaMP5_title":
        first_line = text.splitlines()[0].strip().strip('"')
        first_line = re.sub(r"^(answer|title)\s*:\s*", "", first_line, flags=re.IGNORECASE)
        return first_line or None

    return text


def build_prediction_repair_prompt(task: str, raw_text: str) -> str:
    if task == "LaMP3_rate":
        target = "one integer among 1, 2, 3, 4, or 5; if needed, convert to the nearest valid rating in that range"
    elif task == "LaMP2_tag":
        target = "exactly one tag label"
    elif task == "LaMP5_title":
        target = "exactly one concise title"
    else:
        target = "exactly one final answer"

    return (
        "Your previous answer did not follow the output format.\n"
        f"Previous answer: {raw_text!r}\n"
        f"Return only {target}. Do not output N/A, explanations, or extra text."
    )
