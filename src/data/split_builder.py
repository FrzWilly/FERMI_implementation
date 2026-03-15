from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from FERMI.src.data.io_stream import iter_json_array, load_outputs_map
from FERMI.src.data.lamp_parser import TASKS, UnifiedSample, build_sample


class LAMPSplitBuilder:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def _task_dir(self, task: str) -> Path:
        if task not in TASKS:
            raise ValueError(f"Unsupported task: {task}")
        return self.data_root / task

    def _question_path(self, task: str, split: str) -> Path:
        return self._task_dir(task) / f"{split}_questions.json"

    def _output_path(self, task: str, split: str) -> Path:
        return self._task_dir(task) / f"{split}_outputs.json"

    def load_split(
        self,
        task: str,
        split: str,
        limit: Optional[int] = None,
        check_consistency: bool = True,
    ) -> List[UnifiedSample]:
        q_path = self._question_path(task, split)
        if not q_path.exists():
            raise FileNotFoundError(f"Questions file not found: {q_path}")

        outputs: Dict[str, object] = {}
        o_path = self._output_path(task, split)
        if o_path.exists():
            outputs = load_outputs_map(o_path)

        samples: List[UnifiedSample] = []
        seen_ids = set()

        for record in iter_json_array(q_path):
            sid = str(record["id"])
            if sid in seen_ids:
                raise ValueError(f"Duplicate question id in {q_path}: {sid}")
            seen_ids.add(sid)

            sample = build_sample(record=record, task=task, split=split, gold=outputs.get(sid))
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break

        if check_consistency:
            # 當 limit 啟用時，questions 可能只讀到子集合；
            # 此時 outputs 出現「extra ids」屬正常現象，不應視為錯誤。
            strict_output_match = limit is None
            self._check_consistency(
                split=split,
                question_ids=seen_ids,
                output_ids=set(outputs.keys()),
                strict_output_match=strict_output_match,
            )

        return samples

    @staticmethod
    def group_by_user(samples: List[UnifiedSample]) -> Dict[str, List[UnifiedSample]]:
        groups: Dict[str, List[UnifiedSample]] = defaultdict(list)
        for sample in samples:
            groups[sample.user_id].append(sample)
        return dict(groups)

    @staticmethod
    def _check_consistency(
        split: str,
        question_ids: set,
        output_ids: set,
        strict_output_match: bool = True,
    ) -> None:
        # test split 可以沒有 outputs
        if split == "test" and not output_ids:
            return
        if not output_ids:
            raise ValueError(f"Expected outputs for split={split}, but found none.")

        missing = question_ids - output_ids
        extra = output_ids - question_ids
        if missing:
            raise ValueError(f"Missing gold outputs for {len(missing)} question ids.")
        if strict_output_match and extra:
            raise ValueError(f"Found {len(extra)} extra outputs not present in questions.")
