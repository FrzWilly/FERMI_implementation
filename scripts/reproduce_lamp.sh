#!/usr/bin/env bash
set -euo pipefail

TASKS=(LaMP2_tag LaMP3_rate LaMP5_title)
METHODS=(uniform vanilla fewshot_bm25 fewshot_cont opro fermi)

SPLIT="${1:-test}"
SEED="${2:-42}"
OUT_DIR="${3:-FERMI/results}"

if [[ "${SPLIT}" != "test" && "${SPLIT}" != "dev" && "${SPLIT}" != "train" ]]; then
  echo "[ERROR] split 必須是 test/dev/train 其中之一，當前為: ${SPLIT}" >&2
  exit 1
fi

echo "[INFO] Running LaMP matrix: split=${SPLIT}, seed=${SEED}, output_dir=${OUT_DIR}"

for task in "${TASKS[@]}"; do
  for method in "${METHODS[@]}"; do
    python3 -m FERMI.src.runner.run_method \
      --task "${task}" \
      --method "${method}" \
      --split "${SPLIT}" \
      --seed "${SEED}" \
      --output_dir "${OUT_DIR}"
  done
done

echo "[DONE] Completed 3x6 matrix for split=${SPLIT}."
