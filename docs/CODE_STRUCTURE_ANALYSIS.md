# FERMI Code Structure Analysis

更新時間：2026-03-29（Asia/Taipei）

## 1. 專案現況摘要

[`FERMI/`](FERMI) 目前是一個以 LaMP 三任務為核心的可執行研究骨架，狀態偏向「paper-style reconstruction + 可觀測化強化」：

- 方法面：已具備 baseline、[`OPROMethod`](FERMI/src/methods/opro.py:22) 與 [`FERMIMethod`](FERMI/src/methods/fermi.py:18) 的 per-user 執行流程。
- 結構面：以 [`run_per_user()`](FERMI/src/runner/run_method.py:81) 為唯一主流程，設定採 shared / experiment / method 疊加。
- 資料面：倉內目前主要保存多種 LaMP reconstruction / validation snapshot，而不是單一官方資料根目錄。
- 可靠性面：RoP 與 dense retrieval 都有 fallback；訓練與推論 artifacts 會寫入 summary / curves / snapshots。
- 論文近似設定：本次已對齊為 evaluator M = gpt-3.5-turbo、optimizer Mopt = gpt-4。

## 2. 目錄盤點與責任分工

### 2.1 設定層 [`FERMI/configs/`](FERMI/configs)

| 路徑 | 角色 | 備註 |
|---|---|---|
| [`FERMI/configs/shared/runtime.yaml`](FERMI/configs/shared/runtime.yaml) | 執行期共用設定 | seed / 輸出 / 基本 runtime 參數 |
| [`FERMI/configs/shared/models.yaml`](FERMI/configs/shared/models.yaml) | 共用模型設定 | OpenAI key env、timeout、M / Mopt 預設值 |
| [`FERMI/configs/shared/metrics.yaml`](FERMI/configs/shared/metrics.yaml) | task 對應 metric | evaluator 聚合依此決定 |
| [`FERMI/configs/experiments/`](FERMI/configs/experiments) | task-specific 參數 | `tau`、task 名稱、資料細節 |
| [`FERMI/configs/methods/`](FERMI/configs/methods) | method-specific 參數 | `K/L/T`、RoP、retriever、模型覆寫 |

設定載入順序由 [`_default_config_paths()`](FERMI/src/runner/run_method.py:22) 決定，且採 Python dict 的「後載入覆蓋前載入」語意：

1. [`FERMI/configs/shared/runtime.yaml`](FERMI/configs/shared/runtime.yaml)
2. [`FERMI/configs/shared/models.yaml`](FERMI/configs/shared/models.yaml)
3. [`FERMI/configs/experiments/*.yaml`](FERMI/configs/experiments)
4. [`FERMI/configs/methods/*.yaml`](FERMI/configs/methods)

這代表 method config 會覆蓋 shared model defaults，是目前模型設定的一級耦合點。

### 2.2 資料層 [`FERMI/src/data/`](FERMI/src/data)

| 檔案 | 角色 |
|---|---|
| [`lamp_parser.py`](FERMI/src/data/lamp_parser.py) | 將原始 question/output record 轉為 [`UnifiedSample`](FERMI/src/data/lamp_parser.py:13) |
| [`split_builder.py`](FERMI/src/data/split_builder.py) | 從資料根目錄載入 split、做 question/output 一致性檢查 |
| [`io_stream.py`](FERMI/src/data/io_stream.py) | 以 streaming 方式讀取大型 JSON array |

這一層負責把不同資料來源收斂成統一 sample 格式，並在 [`derive_user_id()`](FERMI/src/data/lamp_parser.py:26) 中用 profile hash 推導 user id。這項設計讓流程可跑，但也使「論文中的 user 定義」與「目前執行期 user 單位」不完全等價。

### 2.3 方法層 [`FERMI/src/methods/`](FERMI/src/methods)

| 檔案 | 角色 |
|---|---|
| [`base.py`](FERMI/src/methods/base.py) | 通用 method 介面、rule-based fallback predictor、profile text 抽象 |
| [`vanilla.py`](FERMI/src/methods/vanilla.py) | 只用 query 做 LLM M 推論 |
| [`fewshot.py`](FERMI/src/methods/fewshot.py) | 先做 retrieval，再組 few-shot prompt 給 LLM M |
| [`opro.py`](FERMI/src/methods/opro.py) | OPRO 的 prompt optimization loop、MemoryBank、LLM 呼叫與 observability |
| [`fermi.py`](FERMI/src/methods/fermi.py) | FERMI 的 mis-aligned context 與 RoP-aware prompt selection |
| [`optimizer_loop.py`](FERMI/src/methods/optimizer_loop.py) | prompt scoring、misaligned records、Figure 8/9 prompt builder |
| [`memory_bank.py`](FERMI/src/methods/memory_bank.py) | top-L prompt pool 管理 |
| [`llm_interface.py`](FERMI/src/methods/llm_interface.py) | OpenAI 呼叫與 missing-key fallback 邏輯 |
| [`observability.py`](FERMI/src/methods/observability.py) | 訓練曲線、memory snapshot、prediction event artifact |

方法繼承主線如下：

- [`BaseMethod`](FERMI/src/methods/base.py:81)
- [`VanillaMethod`](FERMI/src/methods/vanilla.py:14)
- [`FewShotMethod`](FERMI/src/methods/fewshot.py:16)
- [`OPROMethod`](FERMI/src/methods/opro.py:22)
- [`FERMIMethod`](FERMI/src/methods/fermi.py:18)

其中 [`FERMIMethod.fit()`](FERMI/src/methods/fermi.py:35) 明確繞過 [`OPROMethod.fit()`](FERMI/src/methods/opro.py:210)，只呼叫 [`BaseMethod.fit()`](FERMI/src/methods/base.py:88) 儲存 train samples，避免先跑一輪 OPRO 再跑 FERMI 造成雙重優化。這是目前最重要的 method-level 設計決策之一。

### 2.4 檢索層 [`FERMI/src/retrieval/`](FERMI/src/retrieval)

| 檔案 | 角色 |
|---|---|
| [`bm25_retriever.py`](FERMI/src/retrieval/bm25_retriever.py) | lexical / BM25 檢索 |
| [`contriever_retriever.py`](FERMI/src/retrieval/contriever_retriever.py) | Contriever → sentence-transformers → BM25 三層 fallback |
| [`rop_selector.py`](FERMI/src/retrieval/rop_selector.py) | FERMI 的 RoP query-aware prompt selector |

[`RoPSelector`](FERMI/src/retrieval/rop_selector.py:9) 預設要求 MPNet，但若 `sentence-transformers` 不可用或執行期編碼失敗，會切回 lexical 並在 [`summary()`](FERMI/src/retrieval/rop_selector.py:157) 留下 `fallback_reason`。這是目前「論文近似」與「本地可執行性」之間最重要的折衷點。

### 2.5 執行層 [`FERMI/src/runner/`](FERMI/src/runner)

| 檔案 | 角色 |
|---|---|
| [`run_method.py`](FERMI/src/runner/run_method.py) | 單 task × 單 method 主入口 |
| [`run_all.py`](FERMI/src/runner/run_all.py) | 批次跑所有 task / method |
| [`run_task.py`](FERMI/src/runner/run_task.py) | 針對單一 task 批次跑所有 method |

[`run_per_user()`](FERMI/src/runner/run_method.py:81) 是實際核心流程：

1. 依設定鏈載入 config
2. 透過 [`LAMPSplitBuilder.load_split()`](FERMI/src/data/split_builder.py:26) 載入 train 與目標 split
3. 以 [`_group_by_user()`](FERMI/src/runner/run_method.py:74) 逐 user 建 method instance
4. 對 OPRO / FERMI 收集 learning curve
5. 聚合 predictions、metrics、summary 與圖表

此設計將「每個 user 各自 fit」固定為唯一模式，因此不存在全域 pooled optimization 流程。

### 2.6 文件層 [`FERMI/docs/`](FERMI/docs)

| 文件 | 角色 |
|---|---|
| [`ARCHITECTURE.md`](FERMI/docs/ARCHITECTURE.md) | 高層架構、資料流與關鍵設計決策 |
| [`OUTPUT_SPEC.md`](FERMI/docs/OUTPUT_SPEC.md) | 結果檔格式 |
| [`PAPER_GAP_ANALYSIS.md`](FERMI/docs/PAPER_GAP_ANALYSIS.md) | 與論文差距與已修補項目 |
| [`RERUN_PRECHECK.md`](FERMI/docs/RERUN_PRECHECK.md) | rerun 前的模型 / 資料 /成本考量 |
| [`LAMP3_RATE_TIME_PAPER_STYLE.md`](FERMI/docs/LAMP3_RATE_TIME_PAPER_STYLE.md) | time-based paper-style dataset reconstruction 實驗紀錄 |
| [`CODE_STRUCTURE_ANALYSIS.md`](FERMI/docs/CODE_STRUCTURE_ANALYSIS.md) | 本次新增的結構分析總表 |

[`FERMI/docs/SETTINGS_MASTER.md`](FERMI/docs/SETTINGS_MASTER.md) 目前不存在，因此不作更新；相關資訊已併入本文件與 [`ARCHITECTURE.md`](FERMI/docs/ARCHITECTURE.md)。

### 2.7 資料／結果相關目錄

目前倉內與資料或結果最相關的目錄不是單一 [`FERMI/LaMP/`](FERMI/LaMP) 根，而是多組 snapshot / reconstruction：

| 路徑 | 性質 |
|---|---|
| [`FERMI/LaMP_planA_seed42/`](FERMI/LaMP_planA_seed42) | plan-A 取樣資料快照 |
| [`FERMI/LaMP_validation_only_seed42/`](FERMI/LaMP_validation_only_seed42) | validation-only reconstruction |
| [`FERMI/LaMP_time_validation_raw/`](FERMI/LaMP_time_validation_raw) | time-based raw validation inputs |
| [`FERMI/LaMP_user_validation_raw/`](FERMI/LaMP_user_validation_raw) | user-based raw validation inputs |
| [`FERMI/LaMP3_rate_time_paper_style_seed42/`](FERMI/LaMP3_rate_time_paper_style_seed42) | paper-style reconstruction 產物 |
| [`FERMI/results/`](FERMI/results) | 執行時預設輸出位置，依 run 動態建立 |

重點是：runner 預設參數仍使用 `FERMI/LaMP` 作為 canonical data root，但目前倉內實際存在的是多組 reconstruction 目錄，因此使用者執行時通常需要明確提供 `--data_root`。

## 3. 主要資料流

### 3.1 設定資料流

[`run_per_user()`](FERMI/src/runner/run_method.py:81) 先透過 [`load_config_file()`](FERMI/src/utils/io.py:38) 依序讀入 shared / experiment / method 設定，再把最終 config 傳入 method constructor。這使模型值、RoP backend、optimization ratios 都可由 method 檔覆寫 shared defaults。

### 3.2 訓練資料流

1. [`LAMPSplitBuilder.load_split()`](FERMI/src/data/split_builder.py:26) 載入 train split
2. [`build_sample()`](FERMI/src/data/lamp_parser.py:44) 產出 [`UnifiedSample`](FERMI/src/data/lamp_parser.py:13)
3. [`_group_by_user()`](FERMI/src/runner/run_method.py:74) 以 derived user_id 分桶
4. 各 method 執行 [`fit()`](FERMI/src/methods/opro.py:210) 或 [`fit()`](FERMI/src/methods/fermi.py:35)

對 OPRO / FERMI 而言，train split 還會進一步在 [`_split_optimization_and_demonstration()`](FERMI/src/methods/opro.py:167) 拆成 optimization 與 demonstration 子集合。

### 3.3 推論資料流

1. 建立 personalized inference prompt：[`build_personalized_inference_prompt()`](FERMI/src/methods/inference_utils.py:86)
2. 用 [`LLMClient.generate()`](FERMI/src/methods/llm_interface.py:60) 呼叫 evaluator M
3. 解析輸出：[`parse_llm_prediction()`](FERMI/src/methods/inference_utils.py:122)
4. 寫出 prediction event、summary 與 metrics

FERMI 另外在推論前插入 [`RoPSelector.select()`](FERMI/src/retrieval/rop_selector.py:126)；Few-shot 會先經過 retriever 決定 examples。

## 4. 已知耦合點

### 4.1 設定覆蓋耦合

shared model config 與 method config 都能定義 `model_M_name` / `model_Mopt_name`。若兩處不一致，最後以 method 檔為準。這也是本次需要同步更新 [`FERMI/configs/shared/models.yaml`](FERMI/configs/shared/models.yaml)、[`FERMI/configs/methods/opro.yaml`](FERMI/configs/methods/opro.yaml) 與 [`FERMI/configs/methods/fermi.yaml`](FERMI/configs/methods/fermi.yaml) 的原因。

### 4.2 data_root 與實際資料目錄耦合

[`run_method.build_arg_parser()`](FERMI/src/runner/run_method.py:200) 的 `--data_root` 預設值是 `FERMI/LaMP`，但目前倉內主體資料目錄多為 reconstruction 版本。文件若只寫預設值，容易讓使用者以為該根目錄必然存在。

### 4.3 user_id 推導耦合

[`derive_user_id()`](FERMI/src/data/lamp_parser.py:26) 依 profile hash 建 user id。此作法讓 pipeline 可以執行，但也讓「每 user fit」高度依賴 profile 穩定性，而非資料集原生 user 標註。

### 4.4 OPRO / FERMI 共用基底耦合

[`FERMIMethod`](FERMI/src/methods/fermi.py:18) 繼承自 [`OPROMethod`](FERMI/src/methods/opro.py:22)，因此：

- LLM 設定與 observability 管線共用
- personalized inference prompt 共用
- memory bank 與 optimizer generation 共用

好處是維護成本低；代價是 FERMI 修改時要同時注意 OPRO 的預設行為是否被隱式繼承。

## 5. 目前專案狀態判讀

### 5.1 paper-style reconstruction

目前專案不是單純 reproduction script，而是同時內含多組 paper-style reconstruction：

- [`sample_validation_only_paper_style.py`](FERMI/scripts/sample_validation_only_paper_style.py)
- [`build_lamp3_rate_time_paper_style.py`](FERMI/scripts/build_lamp3_rate_time_paper_style.py)
- [`LAMP3_RATE_TIME_PAPER_STYLE.md`](FERMI/docs/LAMP3_RATE_TIME_PAPER_STYLE.md)

這說明資料集層仍在「盡量逼近論文設定，但非官方原始 split」的狀態。

### 5.2 observability 已是核心能力

[`TrainingArtifacts`](FERMI/src/methods/observability.py:120) 會寫出：

- prompt evaluation records
- iteration summary
- memory snapshot
- final memory
- prediction events
- learning curve SVG

因此 FERMI 專案目前不只是能跑方法，也能追蹤 prompt optimization 過程。

### 5.3 RoP / MPNet fallback 已產品化

[`RoPSelector`](FERMI/src/retrieval/rop_selector.py:9) 的 `backend_requested`、`backend_effective`、`fallback_reason` 都會進入 runtime summary，代表 MPNet 不可用已不是異常路徑，而是明確設計的 degrade path。

### 5.4 推論階段仍以「LLM 必須成功」為主

對 [`VanillaMethod`](FERMI/src/methods/vanilla.py:14)、[`FewShotMethod`](FERMI/src/methods/fewshot.py:16)、[`OPROMethod`](FERMI/src/methods/opro.py:22)、[`FERMIMethod`](FERMI/src/methods/fermi.py:18) 而言，正式 predict 階段若 API 失敗會直接 raise，而不是默默切回 rule-based prediction。這保留了研究設定的可觀測性，但也讓外部 API 依賴更明顯。

## 6. 本次結論

FERMI 目前的 code structure 已相對清楚，可概括為：

- runner 負責 per-user orchestration
- data 層負責統一 sample 與 split consistency
- methods 層負責 baseline / OPRO / FERMI 核心研究邏輯
- retrieval 層負責 few-shot 與 RoP query-aware selection
- docs 層承擔 paper-gap、architecture、reconstruction 與 rerun precheck 說明

本次同步完成的關鍵一致化動作是：

- 把模型設定改為 evaluator M = gpt-3.5-turbo、optimizer Mopt = gpt-4
- 補上結構分析文件
- 更新架構與 README，使其反映目前「paper-style reconstruction + observability + fallback-aware」的專案狀態
