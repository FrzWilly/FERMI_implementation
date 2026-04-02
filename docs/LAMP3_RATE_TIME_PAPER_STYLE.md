# LaMP3_rate time-based paper-style dataset 與 baseline 實驗摘要

更新時間：2026-03-28（Asia/Taipei）

## 1. 範圍

本次只處理 [`LaMP3_rate`](../configs/experiments/lamp3_rate.yaml)，資料來源為 [`FERMI/LaMP_time_validation_raw/LaMP3_rate/dev_questions.json`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json) 與 [`FERMI/LaMP_time_validation_raw/LaMP3_rate/dev_outputs.json`](../LaMP_time_validation_raw/LaMP3_rate/dev_outputs.json)。

未修改任何 baseline 演算法核心邏輯；只新增資料重建腳本、資料集輸出、manifest / sampling records，以及在新資料集上的 baseline run 結果。

## 2. 原始資料盤點結論

- 原始 question 數：`2500`
- 原始 top-level [`user_id`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json:1549) 數：`2500`
- 每筆 raw question 都對應一個獨立 top-level user record；因此無法用 raw questions 直接分組出「同一 user 有 50 個 questions」的官方結構。
- 但每筆 raw question 的 [`profile`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json:5) 內含歷史 review items，且每個歷史 item 都帶有 [`date`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json:10) 與分數，可作為 time-based 重建依據。

本次可執行重建採用的 user 單位為：**每筆 raw question 視為一個可重建 user episode**，其歷史來自該筆 question 的 [`profile`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json:5)。

## 3. 資料重建規則

重建腳本：[`FERMI/scripts/build_lamp3_rate_time_paper_style.py`](../scripts/build_lamp3_rate_time_paper_style.py)

對每筆 raw question：

1. 取其 [`profile`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json:5) 歷史 items。
2. 依歷史 item 的 `date`、`id` 進行時間排序，實作在 [`_sort_history_items()`](../scripts/build_lamp3_rate_time_paper_style.py:28)。
3. 要形成 `20 train + 30 test`，需至少 `49` 個歷史 items，因 test 的最後 1 筆保留給官方 target。
4. 若歷史長度足夠，取**最近的 49 個歷史 items**：
   - 前 `20` 個作為 train
   - 後 `29` 個作為 test history
   - 再把官方 target 當成第 `30` 個 test item
5. 每一筆重建後 sample 的 [`profile`](../scripts/build_lamp3_rate_time_paper_style.py:90) 只保留該 50-step sequence 中、時間上更早的 items，避免把未來資訊放回歷史。

核心規則已寫入 [`FERMI/LaMP3_rate_time_paper_style_seed42/dataset_manifest.json`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json)，其中：

- [`selection_policy.min_history_required`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json:13) = `49`
- [`selection_policy.history_window`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json:19) = `use_most_recent_49_history_items_before_official_target`
- [`selection_policy.split_rule`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json:20) = `20_selected_history_items_for_train + 29_later_history_items_for_test + official_target_as_final_test_item`

## 4. 最終資料集輸出

新資料集根目錄：[`FERMI/LaMP3_rate_time_paper_style_seed42/`](../LaMP3_rate_time_paper_style_seed42)

主要檔案：

- [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_questions.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_questions.json)
- [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_outputs.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_outputs.json)
- [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_questions.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_questions.json)
- [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_outputs.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_outputs.json)
- [`FERMI/LaMP3_rate_time_paper_style_seed42/dataset_manifest.json`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json)
- [`FERMI/LaMP3_rate_time_paper_style_seed42/sampling_records.json`](../LaMP3_rate_time_paper_style_seed42/sampling_records.json)

資料集統計：

- eligible raw user episodes：`2500`
- seed：`42`
- request user 數：`50`
- 最終抽樣 user 數：`50`
- train：`1000`（`50 × 20`）
- test：`1500`（`50 × 30`）

因此，**這次確實達成 50 users 的可執行 paper-style reconstruction**；但這是基於 raw question episode 的 heuristic reconstruction，不是官方釋出的明確 user-split 檔案。

## 5. 每 user 的 20/30 切法

完整逐 user sampling record 在 [`FERMI/LaMP3_rate_time_paper_style_seed42/sampling_records.json`](../LaMP3_rate_time_paper_style_seed42/sampling_records.json)。

每位 user 都遵守相同切法：

- `train_record_ids`：20 筆較早的 selected history items
- `test_record_ids`：29 筆較晚的 selected history items + 1 筆官方 target
- `official_target_record_id` 永遠是該 user 的最後一筆 test item，例如 [`9000172_49_91743`](../LaMP3_rate_time_paper_style_seed42/sampling_records.json:6)

以下列出 50 位選中 users 的摘要：

| user_id | official_question_id | eligible_history_length | selected_history_start | selected_history_end | split |
|---|---:|---:|---|---|---|
| 9000172 | 91743 | 108 | 2015-03-18 | 2018-06-05 | 20 train + 29 history test + 1 official target |
| 9000774 | 91699 | 371 | 2017-01-18 | 2017-12-11 | 20 train + 29 history test + 1 official target |
| 9000805 | 911158 | 148 | 2016-10-03 | 2018-04-16 | 20 train + 29 history test + 1 official target |
| 9000864 | 911166 | 197 | 2015-06-22 | 2017-12-06 | 20 train + 29 history test + 1 official target |
| 9000919 | 9188 | 105 | 2011-03-17 | 2018-03-09 | 20 train + 29 history test + 1 official target |
| 9001195 | 911149 | 129 | 2016-06-29 | 2018-07-18 | 20 train + 29 history test + 1 official target |
| 9002475 | 91587 | 125 | 2015-06-17 | 2018-07-15 | 20 train + 29 history test + 1 official target |
| 9002717 | 91313 | 129 | 2015-09-05 | 2018-04-28 | 20 train + 29 history test + 1 official target |
| 9002881 | 911068 | 128 | 2015-11-08 | 2017-10-31 | 20 train + 29 history test + 1 official target |
| 9002909 | 91665 | 99 | 2016-02-20 | 2018-04-28 | 20 train + 29 history test + 1 official target |
| 9003087 | 911238 | 156 | 2015-04-16 | 2018-07-24 | 20 train + 29 history test + 1 official target |
| 9003210 | 911862 | 122 | 2015-05-20 | 2017-06-25 | 20 train + 29 history test + 1 official target |
| 9003227 | 91853 | 205 | 2016-12-17 | 2018-02-15 | 20 train + 29 history test + 1 official target |
| 9003478 | 911220 | 106 | 2014-10-13 | 2018-04-03 | 20 train + 29 history test + 1 official target |
| 9003879 | 911476 | 101 | 2015-07-15 | 2018-04-15 | 20 train + 29 history test + 1 official target |
| 9004373 | 911884 | 156 | 2015-10-20 | 2018-02-03 | 20 train + 29 history test + 1 official target |
| 9004852 | 911801 | 482 | 2018-01-22 | 2018-06-18 | 20 train + 29 history test + 1 official target |
| 9004972 | 91353 | 108 | 2015-09-11 | 2018-03-25 | 20 train + 29 history test + 1 official target |
| 9006292 | 91557 | 139 | 2015-03-11 | 2018-07-10 | 20 train + 29 history test + 1 official target |
| 9006812 | 91936 | 104 | 2016-03-24 | 2017-12-28 | 20 train + 29 history test + 1 official target |
| 9006939 | 911278 | 139 | 2016-01-16 | 2018-01-16 | 20 train + 29 history test + 1 official target |
| 9006992 | 911425 | 115 | 2014-08-28 | 2018-07-07 | 20 train + 29 history test + 1 official target |
| 9007098 | 911277 | 178 | 2017-01-16 | 2018-06-15 | 20 train + 29 history test + 1 official target |
| 9007349 | 911372 | 213 | 2016-12-24 | 2018-02-28 | 20 train + 29 history test + 1 official target |
| 9007825 | 911120 | 166 | 2016-07-23 | 2018-07-25 | 20 train + 29 history test + 1 official target |
| 9008438 | 911952 | 153 | 2015-08-16 | 2018-05-16 | 20 train + 29 history test + 1 official target |
| 9008819 | 911524 | 102 | 2014-11-09 | 2018-05-26 | 20 train + 29 history test + 1 official target |
| 9008957 | 911455 | 122 | 2015-10-04 | 2018-02-26 | 20 train + 29 history test + 1 official target |
| 9008963 | 911098 | 103 | 2016-08-07 | 2018-04-16 | 20 train + 29 history test + 1 official target |
| 9010798 | 911565 | 110 | 2014-11-07 | 2018-07-11 | 20 train + 29 history test + 1 official target |
| 9010923 | 911817 | 110 | 2015-12-10 | 2018-07-19 | 20 train + 29 history test + 1 official target |
| 9011038 | 911374 | 104 | 2015-08-25 | 2018-07-02 | 20 train + 29 history test + 1 official target |
| 9011571 | 911102 | 145 | 2015-02-24 | 2018-07-22 | 20 train + 29 history test + 1 official target |
| 9012247 | 911861 | 140 | 2015-04-01 | 2018-06-08 | 20 train + 29 history test + 1 official target |
| 9012326 | 91456 | 134 | 2016-12-20 | 2018-06-25 | 20 train + 29 history test + 1 official target |
| 9013773 | 91550 | 143 | 2015-02-25 | 2018-07-14 | 20 train + 29 history test + 1 official target |
| 9013833 | 911389 | 111 | 2015-05-20 | 2018-06-13 | 20 train + 29 history test + 1 official target |
| 9013877 | 91265 | 128 | 2015-02-28 | 2018-06-16 | 20 train + 29 history test + 1 official target |
| 9014769 | 911810 | 112 | 2015-04-08 | 2018-06-15 | 20 train + 29 history test + 1 official target |
| 9015146 | 911281 | 135 | 2015-10-22 | 2018-04-25 | 20 train + 29 history test + 1 official target |
| 9016659 | 91540 | 113 | 2017-04-05 | 2018-07-25 | 20 train + 29 history test + 1 official target |
| 9017658 | 911230 | 146 | 2015-10-12 | 2018-06-21 | 20 train + 29 history test + 1 official target |
| 9017896 | 91495 | 194 | 2016-10-19 | 2018-07-17 | 20 train + 29 history test + 1 official target |
| 9017910 | 911680 | 100 | 2014-12-06 | 2018-07-31 | 20 train + 29 history test + 1 official target |
| 9018123 | 911180 | 119 | 2015-09-03 | 2018-07-25 | 20 train + 29 history test + 1 official target |
| 9018356 | 91553 | 128 | 2014-09-30 | 2018-06-27 | 20 train + 29 history test + 1 official target |
| 9019272 | 911246 | 183 | 2016-12-19 | 2018-04-10 | 20 train + 29 history test + 1 official target |
| 9019310 | 911119 | 201 | 2017-05-24 | 2018-04-16 | 20 train + 29 history test + 1 official target |
| 9019769 | 91775 | 209 | 2016-11-03 | 2018-06-21 | 20 train + 29 history test + 1 official target |
| 9019799 | 911064 | 133 | 2015-08-04 | 2018-06-07 | 20 train + 29 history test + 1 official target |

## 6. 高保真 paper-faithful 方法實驗

結果根目錄：[`FERMI/results/lamp3_rate_time_paper_style_seed42/`](../results/lamp3_rate_time_paper_style_seed42)

執行方法：

- [`uniform`](../configs/methods/uniform.yaml)
- [`vanilla`](../configs/methods/vanilla.yaml)
- [`fewshot_bm25`](../configs/methods/fewshot_bm25.yaml)
- [`fewshot_cont`](../configs/methods/fewshot_cont.yaml)
- [`opro`](../configs/methods/opro.yaml)
- [`fermi`](../configs/methods/fermi.yaml)

本輪以 [`run_method.py`](../src/runner/run_method.py) 在 `test` split 上執行；train split 的檔案作為各方法 [`fit()`](../src/methods/base.py:23) 的輸入。

### 6.1 已完成 rerun 指標總表（MAE，越低越好）

| Method | MAE | n_samples | n_users | Run |
|---|---:|---:|---:|---|
| **opro** | **0.2447** | 1500 | 50 | [`20260329-124735-lamp3_rate-opro-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42) |
| fewshot_cont | 0.3467 | 1500 | 50 | [`20260329-121940-lamp3_rate-fewshot_cont-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-121940-lamp3_rate-fewshot_cont-seed42) |
| fewshot_bm25 | 0.3553 | 1500 | 50 | [`20260329-120515-lamp3_rate-fewshot_bm25-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-120515-lamp3_rate-fewshot_bm25-seed42) |
| vanilla | 0.5107 | 1500 | 50 | [`20260329-113936-lamp3_rate-vanilla-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-113936-lamp3_rate-vanilla-seed42) |
| uniform | 1.7940 | 1500 | 50 | [`20260329-111227-lamp3_rate-uniform-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-111227-lamp3_rate-uniform-seed42) |

對應 metrics 檔：

- [`opro metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/metrics.json)
- [`uniform metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-111227-lamp3_rate-uniform-seed42/metrics.json)
- [`vanilla metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-113936-lamp3_rate-vanilla-seed42/metrics.json)
- [`fewshot_bm25 metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-120515-lamp3_rate-fewshot_bm25-seed42/metrics.json)
- [`fewshot_cont metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-121940-lamp3_rate-fewshot_cont-seed42/metrics.json)

目前已完成的高保真 rerun 中，最佳方法是 **[`opro`](../configs/methods/opro.yaml)**，test MAE = **`0.2447`**。

### 6.2 [`fermi`](../configs/methods/fermi.yaml) 目前只保留 partial 結果

因使用者要求停止繼續燒 API 成本，因此本輪不再重跑完整 [`fermi`](../configs/methods/fermi.yaml)，改整理前一天未完成但已保留 artifact 的 high-fidelity partial run：[`20260329-202245-lamp3_rate-fermi-seed42/`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42)

已確認狀態：

- 完成 user 數：`36 / 50`
- 已存在 per-user finalized artifact：`36` 個 [`final_memory.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/users/9000172/final_memory.json)
- 尚未產生最終 [`summary.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/summary.json) 與完整 [`metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/metrics.json)
- 以 [`prediction_events.jsonl`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/users/9000172/prediction_events.jsonl) 回推可解析 test 預測 `1052` 筆（理論上 `36 × 30 = 1080`）
- 這 `1052` 筆可解析預測的 partial MAE 約為 `0.25`

此 partial MAE **不是正式 final metric**，因為：

1. 只涵蓋 `36` 個已完成 users；
2. 尚缺 `28` 筆理論上應有、但無法從 event log 穩定回推的 test 預測；
3. runner 未完成最後 aggregate 與 [`metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/metrics.json) 寫入。

### 6.3 目前可引用排名

1. `opro = 0.2447`
2. `fewshot_cont = 0.3467`
3. `fewshot_bm25 = 0.3553`
4. `vanilla = 0.5107`
5. `uniform = 1.7940`
6. `fermi = pending (36/50 users partial MAE ≈ 0.25, not final)`

## 7. OPRO / FERMI 訓練與 artifact 索引

本輪已完成 [`opro`](../configs/methods/opro.yaml) 的高保真 `test` 結果；[`fermi`](../configs/methods/fermi.yaml) 則僅保留到 partial artifact，未取得完整 final metric。

### 7.1 [`opro`](../configs/methods/opro.yaml)（高保真已完成）

- run：[`20260329-124735-lamp3_rate-opro-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42)
- test MAE：`0.2447`
- 模型：`M = gpt-3.5-turbo`、`Mopt = gpt-4`
- evaluation subset：`0`（即不做 cost-driven subset，與論文方向一致）
- best prompt：逐 user 記錄於 [`summary.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/summary.json) 的 `user_summaries`
- prediction source：`1500` 筆皆為 LLM，無 heuristic fallback
- 索引：
  - [`summary.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/summary.json)
  - [`metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/metrics.json)
  - [`aggregate_learning_curve.csv`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/aggregate_learning_curve.csv)
  - [`aggregate_learning_curve.png`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/aggregate_learning_curve.png)
  - [`users/`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/users)

learning curve 摘要：[`aggregate_learning_curve.csv`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/aggregate_learning_curve.csv) 對 50 users 匯總；各 user 的最佳 prompt 多落在 `t10_k*`，表示 final inference 確實使用了優化後 prompt，而非固定初始 prompt。

### 7.2 [`fermi`](../configs/methods/fermi.yaml)（高保真 partial）

- partial run：[`20260329-202245-lamp3_rate-fermi-seed42`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42)
- 模型：依 high-fidelity rerun 規劃，應為 `M = gpt-3.5-turbo`、`Mopt = gpt-4`
- 狀態：只完成 `36 / 50` users，未生成完整 aggregate [`summary.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/summary.json)
- 可回推 partial test 預測：`1052 / 1080` 筆
- partial MAE：約 `0.25`
- RoP 狀態：因本機 [`sentence-transformers`](FERMI/src/retrieval/rop_selector.py:29) / [`torch`](FERMI/src/retrieval/rop_selector.py:31) 與 NumPy 2 相容性及後續 API 連線不穩，導致高保真 rerun 多次中斷，無法穩定完成 `50 users`
- 索引：
  - [`users/`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/users)
  - 例如 [`users/9000172/final_memory.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/users/9000172/final_memory.json)
  - 例如 [`users/9000172/prediction_events.jsonl`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/users/9000172/prediction_events.jsonl)

另有一個較晚的新 run [`20260330-122621-lamp3_rate-fermi-seed42/`](../results/lamp3_rate_time_paper_style_seed42/20260330-122621-lamp3_rate-fermi-seed42) 僅完成 `10` 個 users；使用者已要求停止繼續燒 API 成本，因此不再追跑。

### 7.3 失敗情況

- [`opro`](../configs/methods/opro.yaml) 已完整完成。
- [`fermi`](../configs/methods/fermi.yaml) 在高保真 rerun 過程中遭遇兩類問題：
  1. MPNet / torch / NumPy 2 相容性警告與 runtime 初始化風險；
  2. OpenAI API 連線錯誤，導致 per-user 迭代中途停止，未能穩定完成 `50 users` aggregate。
- 目前 runner 沒有 resume 機制；重新下指令會建立新的 time-stamped run，而不是接續舊目錄。

## 8. 與論文設定仍有差異

雖然本輪已達成 `50 users × (20 train + 30 test)` 的可執行資料規模，且 [`opro`](../configs/methods/opro.yaml) 已完整依高保真設定跑完，但仍與論文聲稱的官方設定存在以下差異：

1. **非官方 user mapping**：目前 raw file 並未提供「50 位 user、每位 user 對應 50 個 questions」的官方切分表；本次是根據每筆 raw question 的歷史 [`profile`](../LaMP_time_validation_raw/LaMP3_rate/dev_questions.json:5) 進行 heuristic reconstruction。
2. **官方 target 日期缺失**：top-level raw question 沒有獨立的 `date` 欄位，因此官方 target 只能被固定放在 sequence 最尾端；此限制已記錄於 [`dataset_manifest.json`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json:82)。
3. **使用最近 49 個歷史 items**：若某 raw user episode 的歷史超過 49 筆，較早歷史會被捨棄；本輪共捨棄 `6206` 筆較早歷史。
4. **synthetic profile**：重建出的每一筆 sample 使用的是 sequence prefix，而不是官方釋出的原始 multi-question user split。
5. **FERMI 尚未完成 full 50-user high-fidelity aggregate**：目前只能報告 [`20260329-202245-lamp3_rate-fermi-seed42/`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42) 的 `36-user` partial 結果，而非正式 final metric。

## 9. 變更檔案

本次新增 / 修改檔案如下：

- 修改 [`FERMI/src/data/lamp_parser.py`](../src/data/lamp_parser.py)
- 新增 [`FERMI/scripts/build_lamp3_rate_time_paper_style.py`](../scripts/build_lamp3_rate_time_paper_style.py)
- 新增 [`FERMI/LaMP3_rate_time_paper_style_seed42/dataset_manifest.json`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json)
- 新增 [`FERMI/LaMP3_rate_time_paper_style_seed42/sampling_records.json`](../LaMP3_rate_time_paper_style_seed42/sampling_records.json)
- 新增 [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_questions.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_questions.json)
- 新增 [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_outputs.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/train_outputs.json)
- 新增 [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_questions.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_questions.json)
- 新增 [`FERMI/LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_outputs.json`](../LaMP3_rate_time_paper_style_seed42/LaMP3_rate/test_outputs.json)
- 新增 [`FERMI/results/lamp3_rate_time_paper_style_seed42/20260327-171730-lamp3_rate-uniform-seed42/metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260327-171730-lamp3_rate-uniform-seed42/metrics.json)
- 新增 [`FERMI/results/lamp3_rate_time_paper_style_seed42/20260327-171736-lamp3_rate-vanilla-seed42/metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260327-171736-lamp3_rate-vanilla-seed42/metrics.json)
- 新增 [`FERMI/results/lamp3_rate_time_paper_style_seed42/20260327-171748-lamp3_rate-fewshot_bm25-seed42/metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260327-171748-lamp3_rate-fewshot_bm25-seed42/metrics.json)
- 新增 [`FERMI/results/lamp3_rate_time_paper_style_seed42/20260327-171808-lamp3_rate-fewshot_cont-seed42/metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260327-171808-lamp3_rate-fewshot_cont-seed42/metrics.json)
- 新增 [`FERMI/results/lamp3_rate_time_paper_style_seed42/20260327-180952-lamp3_rate-opro-seed42/`](../results/lamp3_rate_time_paper_style_seed42/20260327-180952-lamp3_rate-opro-seed42)
- 新增 [`FERMI/results/lamp3_rate_time_paper_style_seed42/20260328-142205-lamp3_rate-fermi-seed42/`](../results/lamp3_rate_time_paper_style_seed42/20260328-142205-lamp3_rate-fermi-seed42)
- 新增 [`FERMI/docs/LAMP3_RATE_TIME_PAPER_STYLE.md`](./LAMP3_RATE_TIME_PAPER_STYLE.md)
