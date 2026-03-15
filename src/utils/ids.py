from __future__ import annotations

import hashlib
from datetime import datetime


def make_run_id(task: str, method: str, seed: int) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{task.lower()}-{method.lower()}-seed{seed}"


def hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]

