from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


TASKS = ["LaMP2_tag", "LaMP3_rate", "LaMP5_title"]
SOURCE_SPLIT = "dev"
REQUESTED_SIZES = {"train": 1000, "test": 1500}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable_subseed(seed: int, task: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}|{task}|{purpose}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _allocate_sizes(source_size: int, requested_train: int, requested_test: int) -> Tuple[int, int, str, str]:
    requested_total = requested_train + requested_test
    if source_size >= requested_total:
        return requested_train, requested_test, "complete", "exact_without_replacement"

    if source_size <= 0:
        return 0, 0, "empty_source", "unavailable"

    scaled_train = round(source_size * requested_train / requested_total)
    scaled_train = max(0, min(source_size, scaled_train))
    scaled_test = source_size - scaled_train
    return scaled_train, scaled_test, "partial_due_to_insufficient_pool", "proportional_without_replacement"


def _subset_questions(source_questions: List[Dict[str, Any]], indices: List[int]) -> List[Dict[str, Any]]:
    return [dict(source_questions[idx]) for idx in indices]


def _subset_outputs(
    source_output_task_name: str | None,
    source_outputs: Dict[str, Any] | None,
    source_questions: List[Dict[str, Any]],
    indices: List[int],
) -> Dict[str, Any] | None:
    if source_outputs is None:
        return None

    golds: List[Dict[str, Any]] = []
    for idx in indices:
        sid = str(source_questions[idx]["id"])
        if sid not in source_outputs:
            raise KeyError(f"Missing gold output for id={sid}")
        golds.append({"id": sid, "output": source_outputs[sid]})

    return {
        "task": source_output_task_name,
        "golds": golds,
    }


def _build_task_candidate(
    task: str,
    source_root: Path,
    output_root: Path,
    record_root: Path,
    seed: int,
    requested_train: int,
    requested_test: int,
) -> Dict[str, Any]:
    q_path = source_root / task / f"{SOURCE_SPLIT}_questions.json"
    o_path = source_root / task / f"{SOURCE_SPLIT}_outputs.json"

    source_questions = _load_json(q_path)
    source_size = len(source_questions)

    source_outputs = None
    source_output_task_name = None
    if o_path.exists():
        out_obj = _load_json(o_path)
        source_output_task_name = out_obj.get("task")
        source_outputs = {str(item["id"]): item.get("output") for item in out_obj.get("golds", [])}

    actual_train, actual_test, status, allocation_policy = _allocate_sizes(
        source_size=source_size,
        requested_train=requested_train,
        requested_test=requested_test,
    )
    actual_total = actual_train + actual_test

    sub_seed = _stable_subseed(seed=seed, task=task, purpose="validation_only_candidate")
    rng = random.Random(sub_seed)
    shuffled_indices = list(range(source_size))
    rng.shuffle(shuffled_indices)
    selected_indices = shuffled_indices[:actual_total]

    train_indices = selected_indices[:actual_train]
    test_indices = selected_indices[actual_train:actual_train + actual_test]

    train_questions = _subset_questions(source_questions, train_indices)
    test_questions = _subset_questions(source_questions, test_indices)
    train_outputs = _subset_outputs(source_output_task_name, source_outputs, source_questions, train_indices)
    test_outputs = _subset_outputs(source_output_task_name, source_outputs, source_questions, test_indices)

    task_output_dir = output_root / task
    _write_json(task_output_dir / "train_questions.json", train_questions)
    _write_json(task_output_dir / "test_questions.json", test_questions)
    if train_outputs is not None:
        _write_json(task_output_dir / "train_outputs.json", train_outputs)
    if test_outputs is not None:
        _write_json(task_output_dir / "test_outputs.json", test_outputs)

    record = {
        "task": task,
        "source_split": SOURCE_SPLIT,
        "seed": seed,
        "sub_seed": sub_seed,
        "status": status,
        "allocation_policy": allocation_policy,
        "requested_sizes": {
            "train": requested_train,
            "test": requested_test,
            "total": requested_train + requested_test,
        },
        "actual_sizes": {
            "train": actual_train,
            "test": actual_test,
            "total": actual_total,
        },
        "source_size": source_size,
        "used_all_source_records": actual_total == source_size,
        "selected_indices": selected_indices,
        "train_indices": train_indices,
        "test_indices": test_indices,
        "selected_ids": [str(source_questions[idx]["id"]) for idx in selected_indices],
        "train_ids": [str(source_questions[idx]["id"]) for idx in train_indices],
        "test_ids": [str(source_questions[idx]["id"]) for idx in test_indices],
        "source_questions_path": str(q_path),
        "source_outputs_path": str(o_path) if o_path.exists() else None,
        "output_task_dir": str(task_output_dir),
    }

    record_path = record_root / "records" / f"{task}_validation_only_sampling.json"
    _write_json(record_path, record)
    record["record_path"] = str(record_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build validation-only LaMP candidate dataset that more closely follows the paper-style setup."
    )
    parser.add_argument("--source_root", default="FERMI/LaMP")
    parser.add_argument("--output_root", default="FERMI/LaMP_validation_only_seed42")
    parser.add_argument("--record_root", default="FERMI/results/validation_only_seed42_sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_n", type=int, default=1000)
    parser.add_argument("--test_n", type=int, default=1500)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    record_root = Path(args.record_root)

    manifest: Dict[str, Any] = {
        "plan": "validation_only_paper_style_candidate",
        "seed": int(args.seed),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "record_root": str(record_root),
        "source_split": SOURCE_SPLIT,
        "requested_sizes": {
            "train": int(args.train_n),
            "test": int(args.test_n),
            "total": int(args.train_n) + int(args.test_n),
        },
        "tasks": {},
    }

    for task in TASKS:
        manifest["tasks"][task] = _build_task_candidate(
            task=task,
            source_root=source_root,
            output_root=output_root,
            record_root=record_root,
            seed=int(args.seed),
            requested_train=int(args.train_n),
            requested_test=int(args.test_n),
        )

    _write_json(output_root / "dataset_manifest.json", manifest)
    _write_json(record_root / "sampling_manifest.json", manifest)

    print(f"[DONE] Wrote candidate dataset to {output_root}")
    print(f"[DONE] Wrote sampling records to {record_root}")


if __name__ == "__main__":
    main()
