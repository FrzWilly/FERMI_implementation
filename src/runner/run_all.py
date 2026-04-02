from __future__ import annotations

import argparse

from FERMI.src.runner.run_method import run_per_user as run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all methods on all LaMP tasks.")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="FERMI/results")
    parser.add_argument("--data_root", default="FERMI/LaMP")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train_limit", type=int, default=None)
    args = parser.parse_args()

    tasks = ["LaMP2_tag", "LaMP3_rate", "LaMP5_title"]
    methods = ["uniform", "vanilla", "fewshot_bm25", "fewshot_cont", "opro", "fermi"]

    for task in tasks:
        for method in methods:
            run(
                argparse.Namespace(
                    task=task,
                    method=method,
                    config=None,
                    seed=args.seed,
                    split=args.split,
                    output_dir=args.output_dir,
                    data_root=args.data_root,
                    limit=args.limit,
                    train_limit=args.train_limit,
                )
            )


if __name__ == "__main__":
    main()
