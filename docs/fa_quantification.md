### `fa_quantification.py` 在整體實驗中的角色說明

本檔案（`official/SO/fa_quantification.py`）扮演「增強型量化模組」的角色，用以擴展與對照 `speechocean_quantification.py` 的基礎流程。其設計重點在於：引入溫度縮放（Temperature-scaled Softmax）、多種熵度量（Shannon / Rényi / Tsallis）、多樣化的聚合策略（mean/median/min/max/prod），並允許外部提供的音素級對齊（`phoneme_segments` 以幀索引表示）作為輸入，進而生成更豐富的逐音素特徵以支援校準、敏感度分析、消融研究與後續的學習式融合。

---

### 核心定位

- **量化沙盒（Quantification Sandbox）**：提供多種可切換的分數與統計量，方便在開發集上做網格搜尋與消融實驗。
- **外部對齊驅動（FA/對齊器友好）**：不強制執行 CTC segmentation；改以已知的 `phoneme_segments (start_frame, end_frame)` 餵入，適合與強制對齊（Forced Alignment, FA）或其他對齊器搭配。
- **不確定性與校準研究的載體**：內建溫度縮放與熵度量，為 posterior 校準與不確定性量化提供基線與實作骨架。

---

### 與 `speechocean_quantification.py` 的關係與差異

- **對齊方式**
  - `speechocean_quantification.py`：內部以 CTC segmentation 將音素序列與音頻對齊，輸出秒級時間（`start_time`, `end_time`）。
  - `fa_quantification.py`：假設資料集中已提供 `phoneme_segments`（逐音素的 `(start_frame, end_frame)`），直接使用，不執行 CTC segmentation。

- **指標集合**
  - 共同／相近：`max_logit`、`logit_margin`（或 mean/margin 變體）、posterior（標準或溫度縮放）。
  - `fa_quantification.py` 其他擴展：`posterior_prob_temp`（T 溫度縮放）、`entropy`、`renyi_entropy`、`tsallis_entropy`、可配置 `aggregation_method`。

- **輸出模式**
  - `speechocean_quantification.py`：輸出逐音素「秒級」時間段與綜合分數 `combined_score`（以 `alpha` 融合 margin 與負對數 posterior），並可寫入 `phoneme_accuracy`（人評）。
  - `fa_quantification.py`：輸出逐音素「幀級」起訖索引與各類不確定性/校準相關特徵，不含 `combined_score` 與人評欄位（可後處理融合）。

---

### 輸入與輸出（I/O）

- 輸入（資料集至少包含）
  - `audio.array`: 波形
  - `cmu_ipa_phonetic_transcription`: 目標音素序列
  - `cmu_ipa_mispronunciation_transcription`: 對應錯誤音素序列（同長度）
  - `phoneme_segments`: `[(start_frame, end_frame), ...]`（與音素序列同長度）

- 主要超參數
  - `temperature`: 溫度縮放 T
  - `aggregation_method`: 熵等逐幀統計的聚合策略（`mean`/`median`/`min`/`max`/`prod`）

- 輸出（CSV 欄位）
  - `uttid, actual_phoneme, mispronounced_phoneme, start_frame, end_frame, posterior_prob_standard, posterior_prob_temp, max_logit, logit_margin, entropy, renyi_entropy, tsallis_entropy, mispronounced`

---

### 在整體實驗管線中的位置

1. 數據集載入與預處理（HuggingFace `load_from_disk`）
2. 模型初始化（Wav2Vec2 Processor/Tokenizer/ForCTC）
3. 由資料集提供的 `phoneme_segments` 直接切片 logits/probs，計算各類分數
4. 將逐音素特徵輸出為 CSV，供後續評估（notebook/腳本）

其可視為 `speechocean_quantification.py` 的「替代/並行量化步驟」。若已具高品質 FA 或外部對齊器產生的片段，`fa_quantification.py` 能更直接地在對齊後的幀範圍內進行指標計算與研究。

---

### 典型使用情境

- **校準研究**：掃描不同 `temperature` 對 posterior 與判別力的影響，觀察 ROC/MCC 最佳門檻的變化。
- **不確定性分析**：引入 `entropy`/`renyi`/`tsallis` 作為特徵，分析其與人評/錯誤率的相關性。
- **對齊方法對比**：同語料比較 CTC 對齊（`speechocean_quantification.py`） vs. 外部對齊（`fa_quantification.py`）對性能的影響。
- **特徵工程與融合**：把 margin/logit/entropy/posterior_temp 等多特徵餵入簡單分類器（如 logistic regression/小型 MLP）學出「最優融合」。
- **消融與敏感度**：替換聚合策略（mean/median/max/...）、關閉/開啟熵或溫度縮放，比較差異。

---

### 優勢與限制

- 優勢
  - 提供溫度縮放與多種熵度量，可做校準與不確定性量化。
  - 支援多種聚合策略，對短片段或變異分布更具彈性。
  - 接受 `phoneme_segments`，方便與 FA/他牌對齊器整合。

- 限制
  - 依賴外部對齊品質（`phoneme_segments` 必須可靠且對齊到模型幀率）。
  - 目前未內建 `combined_score` 與人評欄位，需於下游整合。
  - 熵的 `alpha`（Rényi/Tsallis）在範例中固定（如 0.33），可再參數化與系統性掃描。

---

### 建議用法與評估對接

- 與 `SO_eval/so_evaluation.ipynb` 對接時：
  - 新增讀取 `fa_quantification.py` 的 CSV 解析器（注意欄位與單位是幀而非秒）。
  - 對 `temperature`、`aggregation_method` 做網格搜尋，在開發集以 MCC/ROC AUC 最佳化，固定後於測試集報告成績。
  - 與 `speechocean_quantification.py` 之輸出同框比較，繪製 ROC/PR、箱型/小提琴圖。

- 學習式融合：
  - 以開發集訓練簡單融合器（LR/MLP），特徵可含 `logit_margin`, `max_logit`, `posterior_prob_temp`, `entropy` 等。
  - 做 per-phoneme 或 per-speaker 標準化以消除系統性偏差。

---

### 小結

`fa_quantification.py` 是整體實驗中的「加值量化插件」。它不強制綁定 CTC 對齊，轉而擁抱外部對齊，並引入溫度縮放與熵等不確定性特徵，為校準、消融與融合研究提供彈性。建議將其輸出與 `speechocean_quantification.py` 並行產生、共同評估，以系統化比較兩種路線（CTC 對齊 vs 外部對齊）與多種指標/聚合策略對最終誤讀檢測成效的影響。


