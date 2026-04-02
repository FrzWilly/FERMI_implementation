from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


TASK = "LaMP3_rate"
SOURCE_SPLIT = "dev"
OUTPUT_TASK_NAME = "LaMP_3"
PROMPT_TEMPLATE = (
    "What is the score of the following review on a scale of 1 to 5? review: {review}"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sort_history_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda item: (str(item.get("date") or ""), str(item.get("id") or "")))


def _format_question_input(review_text: str) -> str:
    return PROMPT_TEMPLATE.format(review=str(review_text or ""))


def _profile_prefix(sequence: List[Dict[str, Any]], end_idx: int) -> List[Dict[str, Any]]:
    return [
        {
            "id": str(item["source_item_id"]),
            "text": item["text"],
            "score": str(item["score"]),
            "date": item.get("date"),
        }
        for item in sequence[:end_idx]
    ]


def _build_outputs(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "task": OUTPUT_TASK_NAME,
        "golds": [{"id": str(record["id"]), "output": str(record["output"])} for record in records],
    }


def _prepare_user_sequence(
    raw_question: Dict[str, Any],
    raw_output: str,
    train_per_user: int,
    test_per_user: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    user_id = str(raw_question["user_id"])
    sorted_history = _sort_history_items(list(raw_question.get("profile", []) or []))
    test_history_n = test_per_user - 1
    selected_history_n = train_per_user + test_history_n
    selected_history = sorted_history[-selected_history_n:]
    dropped_history = sorted_history[:-selected_history_n]

    sequence: List[Dict[str, Any]] = []
    for item in selected_history:
        sequence.append(
            {
                "source_kind": "history",
                "source_item_id": str(item["id"]),
                "text": str(item.get("text", "")),
                "score": str(item.get("score", "")),
                "date": item.get("date"),
            }
        )

    sequence.append(
        {
            "source_kind": "official_target",
            "source_item_id": str(raw_question["id"]),
            "text": str(raw_question.get("input", "").split("review:", 1)[-1].strip()),
            "score": str(raw_output),
            "date": raw_question.get("date"),
        }
    )

    records: List[Dict[str, Any]] = []
    train_record_ids: List[str] = []
    test_record_ids: List[str] = []

    for idx, item in enumerate(sequence):
        split = "train" if idx < train_per_user else "test"
        record_id = f"{user_id}_{idx:02d}_{item['source_item_id']}"
        question = {
            "id": record_id,
            "input": _format_question_input(item["text"]),
            "profile": _profile_prefix(sequence, idx),
            "user_id": user_id,
            "date": item.get("date"),
            "source_item_id": item["source_item_id"],
            "source_kind": item["source_kind"],
            "profile_length": idx,
        }
        records.append(
            {
                "id": record_id,
                "question": question,
                "output": item["score"],
                "split": split,
            }
        )
        if split == "train":
            train_record_ids.append(record_id)
        else:
            test_record_ids.append(record_id)

    history_dates = [item.get("date") for item in selected_history if item.get("date")]
    train_dates = [item.get("date") for item in selected_history[:train_per_user] if item.get("date")]
    test_history_dates = [item.get("date") for item in selected_history[train_per_user:] if item.get("date")]

    sampling_record = {
        "user_id": user_id,
        "official_question_id": str(raw_question["id"]),
        "official_target_included_in_test": True,
        "official_target_record_id": test_record_ids[-1],
        "official_target_date": raw_question.get("date"),
        "eligible_history_length": len(sorted_history),
        "selected_history_length": len(selected_history),
        "dropped_earlier_history_length": len(dropped_history),
        "selected_history_start_date": history_dates[0] if history_dates else None,
        "selected_history_end_date": history_dates[-1] if history_dates else None,
        "train_count": train_per_user,
        "test_count": test_per_user,
        "train_source_item_ids": [str(item["id"]) for item in selected_history[:train_per_user]],
        "test_history_source_item_ids": [str(item["id"]) for item in selected_history[train_per_user:]],
        "train_date_range": {
            "start": train_dates[0] if train_dates else None,
            "end": train_dates[-1] if train_dates else None,
        },
        "test_history_date_range": {
            "start": test_history_dates[0] if test_history_dates else None,
            "end": test_history_dates[-1] if test_history_dates else None,
        },
        "train_record_ids": train_record_ids,
        "test_record_ids": test_record_ids,
    }
    return records, sampling_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaMP3_rate time-based paper-style user dataset.")
    parser.add_argument("--source_root", default="FERMI/LaMP_time_validation_raw/LaMP3_rate")
    parser.add_argument("--output_root", default="FERMI/LaMP3_rate_time_paper_style_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user_count", type=int, default=50)
    parser.add_argument("--train_per_user", type=int, default=20)
    parser.add_argument("--test_per_user", type=int, default=30)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    task_root = output_root / TASK

    raw_questions = _load_json(source_root / f"{SOURCE_SPLIT}_questions.json")
    raw_outputs_obj = _load_json(source_root / f"{SOURCE_SPLIT}_outputs.json")
    raw_outputs = {str(item["id"]): str(item.get("output", "")) for item in raw_outputs_obj.get("golds", [])}

    eligible_users: List[Dict[str, Any]] = []
    min_history_required = int(args.train_per_user) + int(args.test_per_user) - 1

    for raw_question in raw_questions:
        history = list(raw_question.get("profile", []) or [])
        raw_output = raw_outputs.get(str(raw_question["id"]))
        if raw_output is None:
            continue
        if len(history) < min_history_required:
            continue
        eligible_users.append(
            {
                "user_id": str(raw_question["user_id"]),
                "question": raw_question,
                "output": raw_output,
                "history_length": len(history),
            }
        )

    eligible_users = sorted(eligible_users, key=lambda item: (item["user_id"], str(item["question"]["id"])))
    rng = random.Random(int(args.seed))
    if len(eligible_users) <= int(args.user_count):
        selected_users = list(eligible_users)
    else:
        selected_users = rng.sample(eligible_users, k=int(args.user_count))
        selected_users = sorted(selected_users, key=lambda item: item["user_id"])

    train_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []
    sampling_records: List[Dict[str, Any]] = []

    for selected in selected_users:
        records, sampling_record = _prepare_user_sequence(
            raw_question=selected["question"],
            raw_output=selected["output"],
            train_per_user=int(args.train_per_user),
            test_per_user=int(args.test_per_user),
        )
        sampling_records.append(sampling_record)
        train_records.extend(record for record in records if record["split"] == "train")
        test_records.extend(record for record in records if record["split"] == "test")

    train_questions = [record["question"] for record in train_records]
    test_questions = [record["question"] for record in test_records]
    train_outputs = _build_outputs(train_records)
    test_outputs = _build_outputs(test_records)

    manifest = {
        "task": TASK,
        "plan": "time_based_paper_style_user_reconstruction",
        "seed": int(args.seed),
        "source_root": str(source_root),
        "source_split": SOURCE_SPLIT,
        "output_root": str(output_root),
        "selection_policy": {
            "requested_user_count": int(args.user_count),
            "eligible_user_count": len(eligible_users),
            "selected_user_count": len(selected_users),
            "achieved_requested_user_count": len(selected_users) == int(args.user_count),
            "min_history_required": min_history_required,
            "user_sampling": "random_sample_without_replacement_after_sort_by_user_id",
            "history_sort": ["date", "id"],
            "history_window": "use_most_recent_49_history_items_before_official_target",
            "split_rule": "20_selected_history_items_for_train + 29_later_history_items_for_test + official_target_as_final_test_item",
            "synthetic_profile_rule": "each reconstructed sample only sees earlier items inside the selected 50-step user sequence",
        },
        "sizes": {
            "train_per_user": int(args.train_per_user),
            "test_per_user": int(args.test_per_user),
            "train_total": len(train_questions),
            "test_total": len(test_questions),
        },
        "selected_user_ids": [selected["user_id"] for selected in selected_users],
        "notes": [
            "Official raw LaMP3_rate questions expose one top-level target per raw user record.",
            "Official target date is unavailable in the current raw file, so only history item dates are used for chronological ordering.",
            "This reconstruction is paper-style and executable, but it remains a heuristic reconstruction rather than an official split release.",
        ],
    }

    _write_json(task_root / "train_questions.json", train_questions)
    _write_json(task_root / "train_outputs.json", train_outputs)
    _write_json(task_root / "test_questions.json", test_questions)
    _write_json(task_root / "test_outputs.json", test_outputs)
    _write_json(output_root / "dataset_manifest.json", manifest)
    _write_json(output_root / "sampling_records.json", sampling_records)

    print(f"[DONE] eligible_users={len(eligible_users)} selected_users={len(selected_users)}")
    print(f"[DONE] train={len(train_questions)} test={len(test_questions)}")
    print(f"[DONE] output_root={output_root}")


if __name__ == "__main__":
    main()
