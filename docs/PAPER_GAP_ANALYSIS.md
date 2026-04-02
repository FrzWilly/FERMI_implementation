# FERMI Paper Gap Analysis（對照 `references/FERMI.pdf`）

本文件針對目前 [`FERMI/src`](FERMI/src)、[`FERMI/configs`](FERMI/configs) 與論文 Appendix / Setup 規格進行 code-level 差異比對，並標註本次子任務修正狀態。

## 對照基準

- 論文：[`references/FERMI.pdf`](references/FERMI.pdf)
- 主要依據：Section 4.1 + Appendix B.2/B.3 + Figure 8 + Figure 9 + Listing 3 + Listing 6
- 本次目標重點：
  - K=4 / L=5 / T=10
  - RoP \~N=3
  - M temperature=0.0；Mopt temperature=1.0
  - tau(title)=0.2，其餘=1.0
  - 每 user 20 train / 30 test（資料既有，流程不改資料）
  - Uopi 80/20 拆分為 optimization / demonstration（每 user 可重現）

---

## A. Prompt 模板差異（Appendix 對齊）

### A1. OPRO Figure 9 p_opt
- 原狀態：[`generate_k_prompts()`](FERMI/src/methods/optimizer_loop.py:176) 以簡化字串組 prompt，未落地 Figure 9 結構。
- 修正：
  - 新增 [`OPRO_FIGURE9_POPT`](FERMI/src/prompts/templates.py:58)
  - 新增 [`build_opro_popt_prompt()`](FERMI/src/methods/optimizer_loop.py:151)
  - [`OPROMethod.fit()`](FERMI/src/methods/opro.py:91) 改為先建 memory block + demonstration block，再送入 Mopt。
- 狀態：**已修正**

### A2. FERMI Figure 8 p_opt
- 原狀態：FERMI 生成 prompt 仍沿用簡化 `seed + misaligned` 組字串。
- 修正：
  - 新增 [`FERMI_FIGURE8_POPT`](FERMI/src/prompts/templates.py:72)
  - 新增 [`build_fermi_popt_prompt()`](FERMI/src/methods/optimizer_loop.py:160)
  - [`FERMIMethod.fit()`](FERMI/src/methods/fermi.py:31) 改為 Figure 8 形式（memory + demos）。
- 狀態：**已修正**

### A3. Few-shot Listing 3
- 原狀態：few-shot 方法僅做 retrieval+聚合，沒有明確落地 Listing 3 模板常數。
- 修正：
  - 新增 [`FEWSHOT_LISTING3`](FERMI/src/prompts/templates.py:48)
  - 新增 [`build_fewshot_listing3_prompt()`](FERMI/src/prompts/formatter.py:12)
  - 在 [`FewShotMethod.predict()`](FERMI/src/methods/fewshot.py:39) 串接模板建構（目前先做可追溯落地）。
- 狀態：**已修正（模板落地）**

### A4. Vanilla（Listing 6 + tag/title minimal adjustment）
- 原狀態：`DEFAULT_TEMPLATES` 為單行簡述，未對應 Listing 6。
- 修正：
  - [`VANILLA_LAMP_RATE_LISTING6`](FERMI/src/prompts/templates.py:20)
  - [`VANILLA_LAMP_TAG_MIN_ADJUST`](FERMI/src/prompts/templates.py:29)
  - [`VANILLA_LAMP_TITLE_MIN_ADJUST`](FERMI/src/prompts/templates.py:39)
  - [`DEFAULT_TEMPLATES`](FERMI/src/prompts/templates.py:89) 改為對應常數。
- 備註：tag/title 為 **minimal adjustment**（論文未提供完整同等 vanilla 原文）。
- 狀態：**已修正（含最小調整註記）**

---

## B. 方法流程差異（OPRO / FERMI）

### B1. OPRO/FERMI 的 p_opt 組裝
- 原狀態：未依 Figure 9/8 的 memory+demos 分段組裝。
- 修正：[`build_opro_popt_prompt()`](FERMI/src/methods/optimizer_loop.py:151)、[`build_fermi_popt_prompt()`](FERMI/src/methods/optimizer_loop.py:160)；呼叫端更新於 [`OPROMethod.fit()`](FERMI/src/methods/opro.py:91)、[`FERMIMethod.fit()`](FERMI/src/methods/fermi.py:31)。
- 狀態：**已修正**

### B2. FERMI memory context：l=1 詳列 vs l!=1 壓縮
- 原狀態：僅保存 `misaligned_context` 純列表，未反映 l=1 / l!=1 不同表示。
- 修正：
  - [`evaluate_prompt_on_samples()`](FERMI/src/methods/optimizer_loop.py:24) 產生 misaligned records/indices
  - [`build_fermi_memory_block()`](FERMI/src/methods/optimizer_loop.py:123) 實作：
    - l=1：列出 failure cases 明細
    - l!=1：列 common misaligned indices + newly mis-aligned count
- 狀態：**已修正**

### B3. Uopi 80/20 每 user 切分
- 原狀態：[`OPROMethod.fit()`](FERMI/src/methods/opro.py:91) 以全域 sample（舊 20%）抽樣，非每 user。
- 修正：
  - 新增 [`_split_optimization_and_demonstration()`](FERMI/src/methods/opro.py:48)
  - 預設 `split_per_user=true`，每 user 按 80/20 分 optimization/demo。
  - FERMI 繼承並使用相同切分。
- 狀態：**已修正**

### B4. RoP backend 可切換、預設 mpnet、安全 fallback 記錄
- 原狀態：[`RoPSelector`](FERMI/src/retrieval/rop_selector.py:8) 只有 lexical overlap。
- 修正：
  - [`RoPSelector.__init__()`](FERMI/src/retrieval/rop_selector.py:10) 支援 `backend`（預設 `mpnet`）
  - 若 `sentence-transformers` 不可用，fallback lexical，並記錄 `fallback_reason`
  - [`RoPSelector.summary()`](FERMI/src/retrieval/rop_selector.py:111) 可寫入 runtime
  - [`FERMIMethod.runtime_summary()`](FERMI/src/methods/fermi.py:95) 納入 rop 狀態。
- 狀態：**已修正**

### B5. RoP 選擇依 test-query 相關子集計分（Eq.7 精神）
- 原狀態：依 prompt 文本與 query overlap 選 prompt，未使用 Uopi 子集分數。
- 修正：
  - [`RoPSelector.select()`](FERMI/src/retrieval/rop_selector.py:76) 先選 top-\~N relevant opinion ids
  - 若 prompt entry 含 `sample_scores`，以該子集平均分數選 prompt；否則退回相似度。
  - [`OPROMethod`](FERMI/src/methods/opro.py:114) / [`FERMIMethod`](FERMI/src/methods/fermi.py:44) memory entry 納入 `sample_scores`。
- 狀態：**已修正（骨架近似 Eq.7）**

---

## C. 設定檔差異（configs）

### C1. K/L/T/N~/溫度/tau
- 原狀態：K/L/T、N~、溫度已大致一致；tau 在 experiments。
- 修正確認：
  - [`FERMI/configs/methods/opro.yaml`](FERMI/configs/methods/opro.yaml)
  - [`FERMI/configs/methods/fermi.yaml`](FERMI/configs/methods/fermi.yaml)
  - [`FERMI/configs/experiments/lamp2_tag.yaml`](FERMI/configs/experiments/lamp2_tag.yaml)
  - [`FERMI/configs/experiments/lamp3_rate.yaml`](FERMI/configs/experiments/lamp3_rate.yaml)
  - [`FERMI/configs/experiments/lamp5_title.yaml`](FERMI/configs/experiments/lamp5_title.yaml)
- 狀態：**已修正**

### C2. 80/20 roles（optimization/demo）
- 原狀態：方法層僅 `optimization_train_ratio=0.2`。
- 修正：方法/experiment 加入 `optimization_train_ratio=0.8`、`demonstration_ratio=0.2`、`split_per_user=true`。
- 狀態：**已修正**

---

## D. 仍存在偏差（本子任務保留）

1. 目前 prediction 核心仍為規則式 [`predict_from_profile()`](FERMI/src/methods/base.py:43)；
   非完整 LLM `M(q; p)` 推論路徑。已加上 query-overlap 最佳化，但仍屬可執行骨架。
2. Few-shot 模板雖已落地（Listing 3），目前主要用於流程對齊與可追溯，
   尚未將最終預測完全改為 LLM prompt 推論。
3. MPNet backend 需要 `sentence-transformers`；未安裝時會自動 fallback lexical（已記錄）。

---

## E. 本次子任務結論

- 針對「Appendix 模板落地 + OPRO/FERMI 主流程結構 + 80/20 user-level 切分 + RoP backend/fallback」
  已完成可執行修正。
- 保留偏差已明確揭露於本文件與後續設定文件，符合「不可捏造論文內容」原則。
