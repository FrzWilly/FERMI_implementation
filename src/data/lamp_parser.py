from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from FERMI.src.utils.ids import hash_text


TASKS = {"LaMP2_tag", "LaMP3_rate", "LaMP5_title"}


@dataclass
class UnifiedSample:
    id: str
    task: str
    split: str
    input_text: str
    profile: List[Dict[str, Any]]
    gold: Optional[Any]
    user_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def derive_user_id(profile: List[Dict[str, Any]]) -> str:
    if not profile:
        return "user_empty"
    profile_ids = sorted(str(p.get("id", "")) for p in profile)
    signature = "|".join(profile_ids)
    return f"user_{hash_text(signature)}"


def profile_item_text(task: str, item: Dict[str, Any]) -> str:
    if task == "LaMP2_tag":
        return str(item.get("description", ""))
    if task == "LaMP3_rate":
        return str(item.get("text", ""))
    if task == "LaMP5_title":
        return f"{item.get('title', '')} {item.get('abstract', '')}".strip()
    return ""


def build_sample(record: Dict[str, Any], task: str, split: str, gold: Optional[Any]) -> UnifiedSample:
    profile = record.get("profile", []) or []
    return UnifiedSample(
        id=str(record["id"]),
        task=task,
        split=split,
        input_text=str(record.get("input", "")),
        profile=profile,
        gold=gold,
        user_id=derive_user_id(profile),
    )

