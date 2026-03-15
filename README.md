# FERMI + LaMP 重現說明（3 Tasks × 6 Methods）

本目錄提供 LaMP 三任務與 baseline/FERMI 比較流程，重點包含：

- 方法：`uniform`、`vanilla`、`fewshot_bm25`、`fewshot_cont`、`opro`、`fermi`
- 任務：`LaMP2_tag`（Acc）、`LaMP3_rate`（MAE）、`LaMP5_title`（Rouge-L）
- 每次 run 固定輸出：`predictions.json`、`metrics.json`、`summary.json`

## 一鍵重現

### 1) test split（主任務）

```bash
bash FERMI/scripts/reproduce_lamp.sh test 42 FERMI/results
```

### 2) dev split（可計算比較指標）

```bash
bash FERMI/scripts/reproduce_lamp.sh dev 42 FERMI/results
```

## 直接使用 CLI（等價）

```bash
python3 -m FERMI.src.runner.run_all --split test --seed 42 --output_dir FERMI/results --data_root FERMI/LaMP
python3 -m FERMI.src.runner.run_all --split dev --seed 42 --output_dir FERMI/results --data_root FERMI/LaMP
```

## OpenAI API 設定（OPRO / FERMI）

目前 `opro` / `fermi` 的 prompt generation 已支援直接呼叫 OpenAI（新版 SDK `OpenAI` client）。

### 1) 安裝相依套件

```bash
python3 -m pip install -r FERMI/requirements.txt
```

### 2) 設定 API Key

```bash
export OPENAI_API_KEY="sk-..."
```

> 讀取環境變數名稱由 `openai_api_key_env` 控制，預設是 `OPENAI_API_KEY`。

### 3) 模型設定欄位

- `model_M_name`：Evaluator 模型名稱（預設 `gpt-4.1-mini`）
- `model_Mopt_name`：Optimizer 模型名稱（預設 `gpt-4o-mini`）
- `M_temperature`：Evaluator 溫度（目前流程保留）
- `Mopt_temperature`：Optimizer 溫度（實際用於 prompt generation）
- `openai_max_retries`：API 失敗重試次數
- `openai_request_timeout`：API timeout 秒數

可在 [`FERMI/configs/shared/models.yaml`](FERMI/configs/shared/models.yaml) 或方法設定檔覆寫（如 [`FERMI/configs/methods/opro.yaml`](FERMI/configs/methods/opro.yaml)）。

### 4) fallback 條件（明確可控）

- **僅當未設定 key（`OPENAI_API_KEY` 缺失）時**，`LLMClient` 才使用本地 fallback prompt 變體。
- 若有設定 key 但 API 失敗，流程會拋出錯誤，不會靜默改 fallback。
- `summary.json` 的 `method_runtime.llm` 會記錄：
  - `provider`（`openai` 或 `fallback`）
  - `has_api_key`
  - `fallback_reason`
  - `api_calls` / `fallback_calls`

## 結果檔位置

- 單次 run：`FERMI/results/<run-id>/`
  - `FERMI/results/<run-id>/predictions.json`
  - `FERMI/results/<run-id>/metrics.json`
  - `FERMI/results/<run-id>/summary.json`

## 比較表索引

- 設定總表與整合結果：`FERMI/docs/SETTINGS_MASTER.md`
- 輸出格式規範：`FERMI/docs/OUTPUT_SPEC.md`

## 預設論文關鍵參數（已對齊）

- `K=4`、`L=5`、`T=10`
- `rop_n_tilde=3`
- `M_temperature=0.0`
- `Mopt_temperature=1.0`
- `tau(LaMP5_title)=0.2`，其餘 task `tau=1.0`

## 已知 fallback 與 test 指標說明

- `fewshot_cont` 目前為 lexical fallback（`ContrieverRetriever` 介面保留）。
- `opro` / `fermi` 若未設定 `OPENAI_API_KEY`，會 fallback 本地 prompt 變體生成（並於 summary 記錄）。
- 目前資料下 `test` split 無 gold（缺 `test_outputs.json`），因此 `metrics.json.metric_value = null`；
  但 3×6 test run 仍完整產出 `predictions.json`、`metrics.json`、`summary.json`。
