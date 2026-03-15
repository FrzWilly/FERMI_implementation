# FERMI Phase 1 架構說明

## 1. 模組分層

- `src/data/`: LaMP 載入、統一樣本格式、split 與一致性檢查
- `src/retrieval/`: lexical retriever 與 RoP query-aware prompt selection
- `src/methods/`: baseline + OPRO + FERMI 方法迴圈
- `src/eval/`: Accuracy / MAE / Rouge-L 與統一 evaluator
- `src/runner/`: CLI 與批次執行入口

## 2. 執行流程

1. 讀取 config 與 seed
2. 載入 train split（方法訓練/優化）
3. 載入目標 split（dev/test）
4. 進行方法 `fit` 與 `predict`
5. 輸出 predictions / metrics / summary

## 3. FERMI 核心（本階段骨架）

- prompt scoring：以 task metric 轉為 prompt 分數
- mis-aligned context：收集預測錯誤樣本上下文
- memory top-L：保留最高分 prompt 記憶
- generate K prompts：API 可用時呼叫，否則 fallback 生成
- RoP selection：query-aware，預設 lexical/TF-IDF fallback

## 4. 可替換點

- `src/methods/llm_interface.py`: 遠端生成 API
- `src/retrieval/contriever_retriever.py`: dense retrieval
- `src/retrieval/rop_selector.py`: similarity 與 ANN backend

