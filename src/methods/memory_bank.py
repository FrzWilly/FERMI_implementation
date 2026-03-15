from __future__ import annotations

from typing import Dict, List


class MemoryBank:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self.entries: List[Dict] = []

    def add(self, entry: Dict) -> None:
        self.entries.append(entry)
        self.entries.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        self.entries = self.entries[: self.size]

    def top(self) -> List[Dict]:
        return list(self.entries)

    def best(self) -> Dict:
        return self.entries[0] if self.entries else {"prompt_id": "none", "text": "", "score": 0.0}

