# RERUN 前置確認報告（僅設定修正，不重跑全矩陣）

更新時間：2026-03-17（Asia/Taipei）

## 1) 論文設定確認（LaMP 規模、M / Mopt）

依據 [`references/FERMI.pdf`](references/FERMI.pdf) 內文：

- LaMP（tag / rate / title）為 **從原始 validation split 下採樣** 而得，目標規模為：
  - 每個資料集 **train 1,000**、**test 1,500**、**50 users**。
- 模型設定語意：
  - 評估模型 `M`：文中主表以 ChatGPT（gpt-3.5-turbo-0613）為主。
  - 優化模型 `Mopt`：**固定使用 GPT-4**（文中強調 prompt optimization 需較強推理能力）。
- 其他關鍵超參數（FERMI/OPRO）：`K=4`, `L=5`, `T=10`；且優化資料切分約 `80%`（optimization）/ `20%`（demonstration）。
- `tau`：LaMP_title 為 `0.2`，其餘（LaMP_tag / LaMP_rate）為 `1.0`。

## 2) 本地資料規模盤點（LaMP2/3/5）

資料來源目錄：
- [`FERMI/LaMP/LaMP2_tag`](FERMI/LaMP/LaMP2_tag)
- [`FERMI/LaMP/LaMP3_rate`](FERMI/LaMP/LaMP3_rate)
- [`FERMI/LaMP/LaMP5_title`](FERMI/LaMP/LaMP5_title)

### 2.1 question / outputs 數量

| Dataset | train questions | dev questions | test questions | train outputs (`golds`) | dev outputs (`golds`) |
|---|---:|---:|---:|---:|---:|
| LaMP2_tag | 3,820 | 692 | 870 | 3,820 | 692 |
| LaMP3_rate | 20,000 | 2,500 | 2,500 | 20,000 | 2,500 |
| LaMP5_title | 9,682 | 2,500 | 2,500 | 9,682 | 2,500 |

備註：train/dev 的 `outputs.golds` 與 `questions` 數量一致。

### 2.2 可辨識 user 數（若欄位可得）

本地資料的 question record 無明確 `user_id` 欄位；程式目前以 profile 內容雜湊生成 user id（見 [`derive_user_id()`](FERMI/src/data/lamp_parser.py:26)）。

依此邏輯統計結果：

| Dataset | train users (derived) | dev users (derived) | test users (derived) | all users (derived) |
|---|---:|---:|---:|---:|
| LaMP2_tag | 3,820 | 692 | 870 | 5,382 |
| LaMP3_rate | 20,000 | 2,500 | 2,500 | 25,000 |
| LaMP5_title | 9,682 | 2,500 | 2,500 | 14,682 |

此結果顯示「derived user 幾乎與樣本一對一」，與論文的 50 users 設定明顯不一致，表示 **目前資料格式無法可靠還原論文定義的 user 單位**。

## 3) 已修改設定清單（改為論文近似版模型組合）

### 3.1 模型設定修正

1. [`FERMI/configs/shared/models.yaml`](FERMI/configs/shared/models.yaml)
   - `model_M_name`: `gpt-4.1-mini` → `gpt-3.5-turbo`
   - `model_Mopt_name`: `gpt-4o-mini` → `gpt-4`

2. [`FERMI/configs/methods/fermi.yaml`](FERMI/configs/methods/fermi.yaml)
   - `model_M_name`: `gpt-4.1-mini` → `gpt-3.5-turbo`
   - `model_Mopt_name`: `gpt-4o-mini` → `gpt-4`

3. [`FERMI/configs/methods/opro.yaml`](FERMI/configs/methods/opro.yaml)
   - `model_M_name`: `gpt-4.1-mini` → `gpt-3.5-turbo`
   - `model_Mopt_name`: `gpt-4o-mini` → `gpt-4`

### 3.2 對齊說明（語意映射）

- 論文語意是「`Mopt` 需比 `M` 更偏向強推理」；本地配置改為 `gpt-4`（Mopt）對 `gpt-3.5-turbo`（M）做近似映射。
- 這是 **paper-style 近似對齊**，不等同於論文原始 `gpt-3.5-turbo-0613` / 論文時點 GPT-4 版本的完全重現。

## 4) OpenAI API 可用性最小測試建議

- 建議測試方式：單次 `chat.completions` 呼叫，優先測 evaluator M=`gpt-3.5-turbo` 或 optimizer Mopt=`gpt-4`。
- 本文件目前僅記錄設定與前置檢查建議，不假設本次已重新執行外部 API 呼叫。

## 5) 是否存在「資料規模不確定」問題

**存在。**

原因：
1. 論文 LaMP 目標規模為 `1000/1500/50 users`；本地三個資料集規模與此差異很大。
2. 本地資料缺乏可直接對應論文 user 定義的明確欄位（如固定 user_id）。
3. 以 profile 雜湊推導之 user 幾乎與樣本一一對應，顯示目前格式下無法確認是否為論文同一種抽樣單位。

## 6) 在不確定條件下的成本/效果平衡抽樣建議（供決策）

### 建議 A（偏論文可比、低成本）
- 目標：接近論文訓練量級。
- 建議：各 task 先固定 `train_limit=1000`；`dev` 全量；`test` 用現有全量（LaMP2 只有 870，無法到 1500）。
- 優點：成本低、可快速驗證設定與方法趨勢。

### 建議 B（平衡成本與穩定性）
- 目標：較穩定估計，同時控制費用。
- 建議：
  - LaMP2_tag: `train_limit=2000`
  - LaMP3_rate: `train_limit=4000`
  - LaMP5_title: `train_limit=3000`
  - `dev/test` 全量
- 優點：比 A 更穩定，成本明顯低於全量。

### 建議 C（全量本地資料）
- 目標：完全反映當前本地資料分布。
- 建議：train/dev/test 全量。
- 風險：與論文可比性最弱，且成本最高。

---

## 結論

目前已完成「前置確認 + 設定修正」且未啟動全矩陣。下一步建議先由使用者在 A/B/C 抽樣策略中定版，再進行 rerun。
