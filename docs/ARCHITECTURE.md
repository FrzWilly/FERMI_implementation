# FERMI 架構說明

## 1. 模組分層

| 路徑 | 職責 |
|---|---|
| `src/data/` | LaMP 載入、[`UnifiedSample`](FERMI/src/data/lamp_parser.py:13) 建模、split 與一致性檢查 |
| `src/retrieval/` | BM25 / Contriever / MPNet / lexical fallback 與 RoP query-aware selection |
| `src/methods/` | baseline（uniform, vanilla, fewshot）+ OPRO + FERMI 優化迴圈 |
| `src/eval/` | Accuracy / MAE / Rouge-L、統一 evaluator、learning curve 圖表 |
| `src/runner/` | CLI 與批次執行入口；[`run_per_user()`](FERMI/src/runner/run_method.py:81) 是唯一主流程 |
| `src/prompts/` | Figure 8/9、Listing 3/6 與初始 prompt 模板 |
| `configs/` | shared / experiments / methods 三層設定，後載入覆蓋前載入 |
| `docs/` | architecture、paper gap、rerun precheck、paper-style reconstruction 與結構分析 |

---

## 2. 設定解析與覆蓋鏈

設定載入順序由 [`_default_config_paths()`](FERMI/src/runner/run_method.py:22) 決定：

1. [`FERMI/configs/shared/runtime.yaml`](FERMI/configs/shared/runtime.yaml)
2. [`FERMI/configs/shared/models.yaml`](FERMI/configs/shared/models.yaml)
3. [`FERMI/configs/experiments/*.yaml`](FERMI/configs/experiments)
4. [`FERMI/configs/methods/*.yaml`](FERMI/configs/methods)

因此模型、RoP backend、optimization ratio 等值最終都以 method config 為最高優先級。

目前論文近似模型設定為：

- evaluator M = `gpt-3.5-turbo`
- optimizer Mopt = `gpt-4`

---

## 3. 執行流程（per-user 強制）

```text
run_method.run() == run_per_user()
      │
      ├─ load config chain
      ├─ load train_split / target_split
      ├─ build UnifiedSample objects
      ├─ group by derived user_id
      │
      └─ for each user_id:
            method = _build_method(...)        # 每 user 獨立實例
            method.set_run_context(user_dir)
            method.fit(user_train_samples)     # 每 user 獨立訓練
            for sample in user_test:
                out = method.predict(sample)   # 正式推論 LLM 失敗直接 raise
            collect curve_rows (OPRO/FERMI only)
      │
      ├─ evaluate (aggregate all predictions)
      ├─ write predictions.json / metrics.json / user_curves.json / summary.json
      └─ plot aggregate_learning_curve.png
```

---

## 4. 方法繼承關係

```text
BaseMethod
├── UniformMethod    rule: most-frequent label from profile
├── VanillaMethod    LLM M, no profile context (Listing 6)
├── FewShotMethod    retrieval → Listing-3 prompt → LLM M
└── OPROMethod       OPRO prompt optimization loop (T iters, K prompts, L memory)
    └── FERMIMethod  FERMI: mis-aligned context + RoP selection
```

---

## 5. FERMI 核心流程

### fit（每 user 獨立執行）

1. [`_split_optimization_and_demonstration()`](FERMI/src/methods/opro.py:167) 做 `80/20` 拆分
2. 進行 `T=10` 次迭代：
   - 用 evaluator M 對候選 prompt 在 optimization split 上評分
   - 由 [`evaluate_prompt_on_samples()`](FERMI/src/methods/optimizer_loop.py:24) 收集 mis-aligned records
   - 透過 [`build_fermi_popt_prompt()`](FERMI/src/methods/optimizer_loop.py:180) 組 Figure 8 風格的 p_opt
   - 用 optimizer Mopt 生成 `K=4` 新 prompt
   - [`MemoryBank`](FERMI/src/methods/memory_bank.py:6) 保留 top-`L=5`
3. 將 `prompt_pool_final` 存在 [`FERMIMethod`](FERMI/src/methods/fermi.py:33)
4. 用 `memory.best()` 做 `best_prompt`

> [`FERMIMethod.fit()`](FERMI/src/methods/fermi.py:35) 只呼叫 [`BaseMethod.fit()`](FERMI/src/methods/base.py:88)，
> 不呼叫 [`OPROMethod.fit()`](FERMI/src/methods/opro.py:210)，避免雙重優化。

### predict（每 sample）

1. 先把 profile 轉成 opinion sample
2. 呼叫 [`RoPSelector.select()`](FERMI/src/retrieval/rop_selector.py:126) 依 query 挑出相關子集
3. 用 opinion subset 對 prompt pool 打分並選最佳 prompt
4. 呼叫 [`LLMClient.generate()`](FERMI/src/methods/llm_interface.py:60) 讓 evaluator M 產生預測
5. 經 [`parse_llm_prediction()`](FERMI/src/methods/inference_utils.py:122) 解析後輸出

---

## 6. Observability 與 artifacts

[`TrainingArtifacts`](FERMI/src/methods/observability.py:120) 是目前核心可觀測化元件，會持續寫出：

- prompt evaluation records
- iteration summaries
- memory snapshots
- final memory
- prediction events
- learning curve SVG / aggregate PNG

因此目前專案狀態不是單純「把方法跑完」，而是能追蹤整個 prompt optimization 與推論決策過程。

---

## 7. 資料與結果目錄現況

目前倉內與資料／結果最相關的目錄包含：

- [`FERMI/LaMP_planA_seed42/`](FERMI/LaMP_planA_seed42)
- [`FERMI/LaMP_validation_only_seed42/`](FERMI/LaMP_validation_only_seed42)
- [`FERMI/LaMP_time_validation_raw/`](FERMI/LaMP_time_validation_raw)
- [`FERMI/LaMP_user_validation_raw/`](FERMI/LaMP_user_validation_raw)
- [`FERMI/LaMP3_rate_time_paper_style_seed42/`](FERMI/LaMP3_rate_time_paper_style_seed42)
- [`FERMI/results/`](FERMI/results)（執行時動態生成）

這代表 FERMI 目前是「包含多種 reconstruction 與 snapshot 的研究工作區」，而不是只指向單一官方資料根目錄。若使用 CLI，常需要手動指定 `--data_root`。

---

## 8. 關鍵設計決策與耦合點

| 項目 | 說明 |
|---|---|
| per-user 強制 | [`run()`](FERMI/src/runner/run_method.py:195) 是 [`run_per_user()`](FERMI/src/runner/run_method.py:81) 的 alias；不存在 pooled global optimization |
| LLM 失敗不靜默 fallback | vanilla / fewshot / opro / fermi 的正式 predict 路徑在 API 失敗時直接 raise |
| optimizer fallback 僅限缺 key 或允許時 | [`LLMClient`](FERMI/src/methods/llm_interface.py:15) 只在缺少 API key 或顯式允許 fallback 時切到本地模板 |
| user_id 推導 | [`derive_user_id()`](FERMI/src/data/lamp_parser.py:26) 依 profile hash 產生；與論文 user 定義不完全等價 |
| fewshot dense fallback | [`ContrieverRetriever`](FERMI/src/retrieval/contriever_retriever.py:20) 會依序退化到 sentence-transformers / BM25 |
| RoP MPNet fallback | [`RoPSelector`](FERMI/src/retrieval/rop_selector.py:9) 預設 mpnet，但會記錄 effective backend 與 fallback reason |
| 設定覆蓋鏈 | model / tau / retrieval backend 可能同時在 shared 與 method 層出現，需同步維護 |

---

## 9. 目前專案狀態

目前 FERMI 可被理解為：

1. **paper-style reconstruction 持續進行中**：已有多個 reconstruction script 與資料快照。
2. **observability 已內建**：summary、curves、memory snapshots 已成標配。
3. **RoP/MPNet fallback 已穩定產品化**：不是臨時補丁，而是正式 runtime 行為。
4. **模型設定已切回論文近似版**：M = gpt-3.5-turbo，Mopt = gpt-4。

若要再往下一步收斂，優先應處理的是資料根目錄一致化與 user 定義的可驗證性，而不是再擴大 benchmark 覆蓋。
