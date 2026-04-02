from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


TASKS = ["LaMP2_tag", "LaMP3_rate", "LaMP5_title"]
SPLIT_TARGETS = {"train": 1000, "dev": 500, "test": 1500}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _stable_subseed(seed: int, task: str, split: str) -> int:
    digest = hashlib.sha256(f"{seed}|{task}|{split}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _sample_indices(source_size: int, target_size: int, seed: int) -> Tuple[List[int], bool]:
    rng = random.Random(seed)
    if target_size <= source_size:
        return rng.sample(list(range(source_size)), target_size), False
    return [rng.randrange(source_size) for _ in range(target_size)], True


def _sample_split(
    task: str,
    split: str,
    source_questions: List[Dict[str, Any]],
    source_outputs: Dict[str, Any] | None,
    output_task_name: str | None,
    target_size: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    source_size = len(source_questions)
    sub_seed = _stable_subseed(seed=seed, task=task, split=split)
    sampled_indices, used_replacement = _sample_indices(source_size=source_size, target_size=target_size, seed=sub_seed)

    sampled_questions: List[Dict[str, Any]] = []
    sampled_golds: List[Dict[str, Any]] = []
    selected_source_ids: List[str] = []
    sampled_ids: List[str] = []
    seen_new_ids = set()
    source_id_counts = defaultdict(int)

    for source_index in sampled_indices:
        source_record = source_questions[source_index]
        source_id = str(source_record["id"])
        source_id_counts[source_id] += 1

        if source_id_counts[source_id] == 1 and source_id not in seen_new_ids:
            sampled_id = source_id
        else:
            sampled_id = f"{source_id}__dup{source_id_counts[source_id]-1}"
            while sampled_id in seen_new_ids:
                source_id_counts[source_id] += 1
                sampled_id = f"{source_id}__dup{source_id_counts[source_id]-1}"

        sampled_record = dict(source_record)
        sampled_record["id"] = sampled_id
        sampled_questions.append(sampled_record)
        selected_source_ids.append(source_id)
        sampled_ids.append(sampled_id)
        seen_new_ids.add(sampled_id)

        if source_outputs is not None:
            sampled_golds.append({"id": sampled_id, "output": source_outputs[source_id]})

    sampled_outputs_obj = None
    if source_outputs is not None:
        sampled_outputs_obj = {
            "task": output_task_name,
            "golds": sampled_golds,
        }

    record = {
        "task": task,
        "split": split,
        "seed": seed,
        "sub_seed": sub_seed,
        "source_size": source_size,
        "target_size": target_size,
        "used_replacement": used_replacement,
        "n_unique_source_indices": len(set(sampled_indices)),
        "n_unique_source_ids": len(set(selected_source_ids)),
        "n_unique_sampled_ids": len(set(sampled_ids)),
        "selected_source_indices": sampled_indices,
        "selected_source_ids": selected_source_ids,
        "sampled_ids": sampled_ids,
    }
    return sampled_questions, sampled_outputs_obj, record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Plan-A sampled LaMP dataset (seed=42, fixed train/dev/test sizes).")
    parser.add_argument("--source_root", default="FERMI/LaMP")
    parser.add_argument("--output_root", default="FERMI/LaMP_planA_seed42")
    parser.add_argument("--record_root", default="FERMI/results/planA_seed42_sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_n", type=int, default=1000)
    parser.add_argument("--dev_n", type=int, default=500)
    parser.add_argument("--test_n", type=int, default=1500)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    record_root = Path(args.record_root)

    targets = {
        "train": int(args.train_n),
        "dev": int(args.dev_n),
        "test": int(args.test_n),
    }

    manifest: Dict[str, Any] = {
        "plan": "A",
        "seed": int(args.seed),
        "targets": targets,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "record_root": str(record_root),
        "tasks": {},
    }

    for task in TASKS:
        task_manifest: Dict[str, Any] = {}
        for split in ["train", "dev", "test"]:
            q_path = source_root / task / f"{split}_questions.json"
            out_path = source_root / task / f"{split}_outputs.json"

            source_questions = _load_json(q_path)

            source_outputs = None
            output_task_name = None
            if out_path.exists():
                out_obj = _load_json(out_path)
                output_task_name = out_obj.get("task")
                source_outputs = {str(item["id"]): item.get("output") for item in out_obj.get("golds", [])}

            sampled_questions, sampled_outputs_obj, record = _sample_split(
                task=task,
                split=split,
                source_questions=source_questions,
                source_outputs=source_outputs,
                output_task_name=output_task_name,
                target_size=targets[split],
                seed=int(args.seed),
            )

            out_task_dir = output_root / task
            _write_json(out_task_dir / f"{split}_questions.json", sampled_questions)

            if sampled_outputs_obj is not None:
                _write_json(out_task_dir / f"{split}_outputs.json", sampled_outputs_obj)

            rec_path = record_root / "records" / f"{task}_{split}_sampling.json"
            _write_json(rec_path, record)
            record["record_path"] = str(rec_path)
            task_manifest[split] = record

        manifest["tasks"][task] = task_manifest

    manifest_path = record_root / "sampling_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"[DONE] Wrote sampled dataset to {output_root}")
    print(f"[DONE] Wrote sampling records to {record_root}")
    print(f"[DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
