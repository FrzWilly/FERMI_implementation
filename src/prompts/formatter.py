from __future__ import annotations

from typing import List

from FERMI.src.data.lamp_parser import UnifiedSample
from FERMI.src.prompts.templates import FEWSHOT_LISTING3


def format_prompt(base_prompt: str, input_text: str, demos: List[str] | None = None) -> str:
    demos = demos or []
    demo_block = "\n".join(demos)
    return f"{base_prompt}\n\n{demo_block}\n\nInput:\n{input_text}\n\nOutput:".strip()


def build_fewshot_listing3_prompt(query: UnifiedSample, retrieved: List[dict]) -> str:
    rows: List[str] = []
    for idx, r in enumerate(retrieved, start=1):
        if query.task == "LaMP2_tag":
            r_question = str(r.get("description", ""))
        elif query.task == "LaMP3_rate":
            r_question = str(r.get("text", ""))
        else:
            r_question = str(r.get("abstract", ""))

        rows.append(f"[{idx}].")
        rows.append(f"Question:{r_question}")
        if query.task == "LaMP2_tag":
            rows.append(
                "Answer choices:A. sci-fi B. based on a book C. comedy D. action E. twist ending "
                "F. dystopia G. dark comedy H. classic I. psychology J. fantasy K. romance "
                "L. thought-provoking M. social commentary N. violence O. true story"
            )
            rows.append(f"Answer:{r.get('tag', '')}")
        elif query.task == "LaMP3_rate":
            rows.append("Answer choices:N/A")
            rows.append(f"Answer:{r.get('score', '')}")
        else:
            rows.append("Answer choices:N/A")
            rows.append(f"Answer:{r.get('title', '')}")

    answer_choices = "N/A"
    if query.task == "LaMP2_tag":
        answer_choices = (
            "A. sci-fi B. based on a book C. comedy D. action E. twist ending "
            "F. dystopia G. dark comedy H. classic I. psychology J. fantasy K. romance "
            "L. thought-provoking M. social commentary N. violence O. true story"
        )

    return FEWSHOT_LISTING3.format(
        retrieved_block="\n".join(rows),
        question=query.input_text,
        answer_choices=answer_choices,
    )
