from __future__ import annotations

from FERMI.src.prompts.templates import DEFAULT_TEMPLATES


def get_initial_prompt(task: str, strategy: str = "vanilla") -> str:
    base = DEFAULT_TEMPLATES.get(task, "You are a helpful assistant.")
    if strategy == "vanilla":
        return base
    if strategy == "uniform":
        return f"{base} Use the most frequent pattern from history."
    return base

