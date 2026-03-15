from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def iter_json_array(path: Path, chunk_size: int = 65536) -> Iterator[dict]:
    """
    串流解析「頂層為 array」的 JSON，避免一次載入全部內容。
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buffer = ""

        # 找到 array 起始 '['
        while True:
            if "[" in buffer:
                idx = buffer.index("[")
                buffer = buffer[idx + 1 :]
                break
            chunk = f.read(chunk_size)
            if not chunk:
                return
            buffer += chunk

        while True:
            buffer = buffer.lstrip()
            if not buffer:
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buffer += chunk
                continue

            if buffer.startswith("]"):
                return

            if buffer.startswith(","):
                buffer = buffer[1:]
                continue

            try:
                obj, idx = decoder.raw_decode(buffer)
                if isinstance(obj, dict):
                    yield obj
                buffer = buffer[idx:]
            except json.JSONDecodeError:
                chunk = f.read(chunk_size)
                if not chunk:
                    # EOF 時若無法 decode，表示格式有問題
                    remainder = buffer.strip()
                    if remainder in {"", "]"}:
                        return
                    raise
                buffer += chunk


def load_outputs_map(path: Path) -> Dict[str, object]:
    """
    輸出檔目前採一般 json 載入（體積通常顯著小於 questions）。
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    golds = obj.get("golds", [])
    mapping: Dict[str, object] = {}
    for item in golds:
        mapping[str(item["id"])] = item.get("output")
    return mapping

