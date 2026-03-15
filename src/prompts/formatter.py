from __future__ import annotations

from typing import List


def format_prompt(base_prompt: str, input_text: str, demos: List[str] | None = None) -> str:
    demos = demos or []
    demo_block = "\n".join(demos)
    return f"{base_prompt}\n\n{demo_block}\n\nInput:\n{input_text}\n\nOutput:".strip()

