# SETTINGS MASTER

本文件為本次 LaMP 三任務（`LaMP2_tag`、`LaMP3_rate`、`LaMP5_title`）與六方法（`uniform`、`vanilla`、`fewshot_bm25`、`fewshot_cont`、`opro`、`fermi`）之單一設定總表與結果彙整。

## 1) 資料設定

- `data_root`: `FERMI/LaMP`
- tasks:
  - `FERMI/LaMP/LaMP2_tag`
  - `FERMI/LaMP/LaMP3_rate`
  - `FERMI/LaMP/LaMP5_title`
- splits:
  - train: 有 `*_questions.json` + `*_outputs.json`
  - dev: 有 `*_questions.json` + `*_outputs.json`
  - test: 僅 `*_questions.json`（目前無 `test_outputs.json`）

## 2) 方法與論文對齊超參數

### 2.1 共用/關鍵設定（對齊論文預設）

- `K = 4`
- `L = 5`
- `T = 10`
- `rop_n_tilde = 3`
- `M_temperature = 0.0`
- `Mopt_temperature = 1.0`
- `tau(LaMP5_title) = 0.2`
- `tau(LaMP2_tag, LaMP3_rate) = 1.0`

### 2.2 task-metric 對應

- `LaMP2_tag` → `accuracy`（越高越好）
- `LaMP3_rate` → `mae`（越低越好）
- `LaMP5_title` → `rouge_l`（越高越好）

## 3) 目前實作設定來源

- 實驗設定：
  - `FERMI/configs/experiments/lamp2_tag.yaml`
  - `FERMI/configs/experiments/lamp3_rate.yaml`
  - `FERMI/configs/experiments/lamp5_title.yaml`
- 方法設定：
  - `FERMI/configs/methods/uniform.yaml`
  - `FERMI/configs/methods/vanilla.yaml`
  - `FERMI/configs/methods/fewshot_bm25.yaml`
  - `FERMI/configs/methods/fewshot_cont.yaml`
  - `FERMI/configs/methods/opro.yaml`
  - `FERMI/configs/methods/fermi.yaml`
- 共用設定：
  - `FERMI/configs/shared/runtime.yaml`
  - `FERMI/configs/shared/models.yaml`
  - `FERMI/configs/shared/metrics.yaml`

## 4) 執行指令（可重現）

### 4.1 單一矩陣執行（建議）

```bash
bash FERMI/scripts/reproduce_lamp.sh test 42 FERMI/results
```

### 4.2 逐一/批次 CLI

```bash
python3 -m FERMI.src.runner.run_all --split test --seed 42 --output_dir FERMI/results --data_root FERMI/LaMP
python3 -m FERMI.src.runner.run_all --split dev --seed 42 --output_dir FERMI/results --data_root FERMI/LaMP
```

## 5) 隨機種子

- 預設與本次重現使用 `seed=42`。

## 6) 輸出規範

每個 run 輸出到：`FERMI/results/<run-id>/`

- `predictions.json`
- `metrics.json`
- `summary.json`

`summary.json` 含完整 resolved config 快照（由 runner 載入與合併後寫入）。

## 7) 已知 fallback 行為

- `fewshot_cont`：目前 `ContrieverRetriever` fallback 到 lexical 檢索（與 BM25-like 同路徑）。
- `opro` / `fermi`：
  - 使用 OpenAI SDK（`OpenAI` client）呼叫 `chat.completions.create` 進行 prompt generation。
  - 若未設定 `OPENAI_API_KEY`（或 `openai_api_key_env` 指向的 key），`LLMClient` 會 fallback 生成可執行 prompt 變體。
  - 若 key 已設定但 API 失敗，流程會拋出錯誤，不會自動靜默 fallback。
  - 每次 run 的 `summary.json` 會記錄 `method_runtime.llm`（`provider`、`has_api_key`、`fallback_reason`、`api_calls`、`fallback_calls`）。
- 由於當前資料夾未提供 `test_outputs.json`，`test` split 會產生預測檔但 `metrics.json.metric_value = null`。

## 10) OpenAI 相關設定欄位

設定來源（runner 依序合併）：

- `FERMI/configs/shared/models.yaml`
- `FERMI/configs/methods/opro.yaml`
- `FERMI/configs/methods/fermi.yaml`

主要欄位：

- `openai_api_key_env`：API key 環境變數名（預設 `OPENAI_API_KEY`）
- `openai_max_retries`：OpenAI 呼叫重試次數
- `openai_request_timeout`：請求逾時秒數
- `model_M_name`：Evaluator 模型（預設 `gpt-4.1-mini`）
- `model_Mopt_name`：Optimizer 模型（預設 `gpt-4o-mini`，可改為 `gpt-4.1-mini` 等）
- `M_temperature`：Evaluator 溫度
- `Mopt_temperature`：Optimizer 溫度

安裝與環境變數：

```bash
python3 -m pip install -r FERMI/requirements.txt
export OPENAI_API_KEY="sk-..."
```

## 8) 本次 3×6 比較結果（可直接引用）

> 說明：目前可計算主指標者為 `dev`；`test` 僅有 predictions（無 gold），故主指標為 `null`。

### 8.1 方法 × 任務主指標（dev）

| Method | LaMP2_tag (Acc ↑) | LaMP3_rate (MAE ↓) | LaMP5_title (Rouge-L ↑) |
|---|---:|---:|---:|
| uniform | 0.0650 | 0.8772 | 0.0545 |
| vanilla | **0.3150** | 0.6580 | 0.0793 |
| fewshot_bm25 | 0.2818 | **0.5764** | **0.2478** |
| fewshot_cont | 0.2818 | **0.5764** | **0.2478** |
| opro | **0.3150** | 0.6580 | 0.0793 |
| fermi | **0.3150** | 0.6580 | 0.0793 |

### 8.2 各任務最佳方法

- `LaMP2_tag`: `vanilla` / `opro` / `fermi` 並列最佳（Acc=0.3150）
- `LaMP3_rate`: `fewshot_bm25` / `fewshot_cont` 並列最佳（MAE=0.5764）
- `LaMP5_title`: `fewshot_bm25` / `fewshot_cont` 並列最佳（Rouge-L=0.2478）

### 8.3 FERMI 相對 baseline 的 delta（dev）

定義：
- Acc / Rouge-L：`delta = fermi - baseline`（正值代表 FERMI 較佳）
- MAE：`delta = baseline - fermi`（正值代表 FERMI 較佳）

| Baseline | Δ LaMP2_tag | Δ LaMP3_rate | Δ LaMP5_title |
|---|---:|---:|---:|
| uniform | +0.2500 | +0.2192 | +0.0248 |
| vanilla | +0.0000 | +0.0000 | +0.0000 |
| fewshot_bm25 | +0.0332 | -0.0816 | -0.1685 |
| fewshot_cont | +0.0332 | -0.0816 | -0.1685 |
| opro | +0.0000 | +0.0000 | +0.0000 |

## 9) test split 完整性狀態

- 3 tasks × 6 methods 已全部完成並輸出三檔：`predictions.json`、`metrics.json`、`summary.json`。
- 目前 `metrics.json.metric_value` 為 `null`（因缺 test gold），屬資料條件限制而非執行失敗。
