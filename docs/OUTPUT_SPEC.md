# OUTPUT SPEC（Phase 1）

每次 run 輸出：

- `predictions.json`
- `metrics.json`
- `summary.json`

`predictions.json` 每筆至少含：

- `id`, `task`, `method`, `split`, `prediction`, `gold`
- `is_correct`, `abs_error`, `rouge_l`
- `selected_prompt_id`, `rop_neighbor_ids`, `user_id`

