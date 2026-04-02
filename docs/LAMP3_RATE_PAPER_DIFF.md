# LaMP3_rate 與 [`FERMI.pdf`](../references/FERMI.pdf) 的目前差異整理

更新時間：2026-03-30（Asia/Taipei）

## 1. 已對齊論文的部分

### 1.1 資料規模目標

- 已重建出 [`50 users × (20 train + 30 test)`](../LaMP3_rate_time_paper_style_seed42/dataset_manifest.json) 的可執行資料集
- 對齊論文在 [`FERMI.pdf`](../references/FERMI.pdf) 中對 [`LaMP_rate`](../references/FERMI.pdf) 的 `1000 train / 1500 test / 50 users` 規模描述

### 1.2 user 內部切法

- 每位 user 的 `20` 筆 train 會再依 [`optimization_train_ratio`](../configs/experiments/lamp3_rate.yaml) = `0.8` 與 [`demonstration_ratio`](../configs/experiments/lamp3_rate.yaml) = `0.2` 切成約 `16 optimization + 4 demonstrations`
- 這與論文的 `80/20` 內部分工一致

### 1.3 方法超參數

- [`OPRO`](../configs/methods/opro.yaml) 與 [`FERMI`](../configs/methods/fermi.yaml) 已對齊：
  - `T = 10`
  - `K = 4`
  - `L = 5`
  - `rop_n_tilde = 3`

### 1.4 模型設定

- shared model config 已回到論文近似設定：[`models.yaml`](../configs/shared/models.yaml)
  - `M = gpt-3.5-turbo`
  - `Mopt = gpt-4`

### 1.5 不再保留 cost-driven evaluation subset

- [`training_eval_subset_size`](../configs/methods/opro.yaml) 與 [`training_eval_subset_size`](../configs/methods/fermi.yaml) 已設成 `0`
- 因此本次不再為了省成本只抽 optimization samples 的子集評分

### 1.6 OPRO 最終推論使用優化後 prompt

- 已完成的高保真 [`OPRO run`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/summary.json) 顯示各 user 的 [`best_prompt_id`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/summary.json) 多為 `t10_k*`
- 表示最終推論並非固定使用 `init_0`，而是確實使用優化後 prompt

## 2. 尚未對齊論文的部分

### 2.1 不是官方 user split

- 目前 [`LaMP3_rate`](../LaMP3_rate_time_paper_style_seed42) 是從 raw validation question episode heuristic reconstruction 而來
- 並非作者釋出的官方 50-user subsampled validation split

### 2.2 FERMI 尚未取得完整 50-user final metric

- 目前高保真 [`FERMI`](../configs/methods/fermi.yaml) rerun 只保留 partial artifact：
  - [`20260329-202245-lamp3_rate-fermi-seed42/`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42) 完成 `36/50` users
  - 可回推 `1052` 筆 test 預測，partial MAE 約 `0.25`
- 但尚未形成完整 [`summary.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/summary.json) / [`metrics.json`](../results/lamp3_rate_time_paper_style_seed42/20260329-202245-lamp3_rate-fermi-seed42/metrics.json)

### 2.3 runner 無 resume 機制

- [`make_run_id()`](../src/utils/ids.py:7) 會產生新的 time-stamped run id
- [`run_per_user()`](../src/runner/run_method.py:81) 不會接續既有未完成 run
- 因此 FERMI 一旦中途中斷，再次啟動會新開 run，而不是接續剩餘 users

### 2.4 第三方依賴與環境穩定性

- [`RoPSelector`](../src/retrieval/rop_selector.py:9) 依賴 `sentence-transformers` / `torch`
- 本機環境出現 NumPy 2 與部分 compiled module 的相容性警告
- 同時高保真 rerun 過程中出現 OpenAI API 連線失敗，造成 FERMI 多次停在 partial state

## 3. 原因分類

### 3.1 資料限制

- raw file 缺乏官方 user mapping
- official target 本身沒有獨立 `date`

### 3.2 模型 / API 可用性

- high-fidelity FERMI 需要大量 `M = gpt-3.5-turbo` 與 `Mopt = gpt-4` 呼叫
- 長時間批次執行時，OpenAI API 連線偶發失敗會直接中斷 run

### 3.3 實作限制

- [`run_method.py`](../src/runner/run_method.py:81) 缺乏 resume / skip-completed-users 機制
- 導致 partial artifact 不能被直接接續完成

### 3.4 第三方依賴

- MPNet / torch / sentence-transformers 與本機 NumPy 2 組合存在風險
- 雖然不一定每次都致命，但會增加 FERMI rerun 的不穩定性

## 4. 對結果可能造成的影響

### 4.1 資料重建誤差

- 因為不是官方 split，最終 MAE 與論文表格不能視為 one-to-one 可比
- 結果更適合解讀為「paper-style reconstruction 上的近似重現」

### 4.2 FERMI final ranking 仍有不確定性

- 高保真 [`OPRO`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/metrics.json) 已完成，MAE = `0.2447`
- 高保真 FERMI 目前只有 `36-user` partial MAE 約 `0.25`
- 因為缺少剩餘 `14` 個 users，無法斷言 FERMI 最終一定優於或劣於 OPRO

### 4.3 成本與時間放大

- 沒有 resume 會讓中斷後重跑成本顯著上升
- 這也是本次使用者要求停止繼續燒 API 成本、直接整理 `36-user` partial run 的主因

## 5. 目前最務實的結論

- 在已完成的高保真 full run 中，最佳可確認結果是 [`OPRO = 0.2447`](../results/lamp3_rate_time_paper_style_seed42/20260329-124735-lamp3_rate-opro-seed42/metrics.json)
- [`FERMI`](../configs/methods/fermi.yaml) 的高保真版本目前只到 `36/50 users` partial，partial MAE 約 `0.25`
- 因此目前與論文最主要的差距，不在 `T/K/L`、模型名或 `16/4` 切法，而在：
  1. **資料不是官方 split**
  2. **FERMI full 50-user rerun 尚未完成**
  3. **runner 缺乏 resume，導致中斷後無法低成本接續**
