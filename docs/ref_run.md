### 你要做什麼（總覽）
- 使用這兩個腳本從語音資料集中抽取逐音素分數，輸出成CSV。
- 二選一：
  - `speechocean_quantification.py`: 自動用 CTC 對齊（不需要外部時間標註）。
  - `fa_quantification.py`: 使用資料集中提供的逐音素幀級對齊（需要 `phoneme_segments`）。

### 先決條件
- Python 3.9+，PyTorch，Transformers，Datasets，ctc_segmentation 等。
- 建議先安裝官方需求：
```bash
pip install -r official/requirements.txt
```

- 預訓練模型（預設）：`facebook/wav2vec2-xlsr-53-espeak-cv-ft`
- 取樣率：16000 Hz
- 有一個 HuggingFace Datasets 形式的資料集（load_from_disk 可載入）

---

### 你的資料需要長什麼樣
- 兩者皆需：
  - `audio`: 包含 `{"array": np.ndarray, "path": str}` 的欄位
  - `uttid`: 話語ID
  - `cmu_ipa_phonetic_transcription`: 真實音素序列（list[str]）
  - `cmu_ipa_mispronunciation_transcription`: 錯誤音素序列（list[str]，與上同長）

- `speechocean_quantification.py` 額外期望（若要寫入人工分數）：
  - `words`: list[dict]，其中每個 `word` 含 `phones-accuracy`（逐音素人工分數）；腳本會用 `map(extract_phoneme_accuracies)` 拉平為 `phoneme_accuracies`

- `fa_quantification.py` 額外必須：
  - `phoneme_segments`: list[tuple(int start_frame, int end_frame)]，與音素序列同長（幀級對齊）

把這兩者想像成兩種評審模式：
- speechocean：智能攝影師，自己對焦（自動切音素）
- fa：精密手錶匠，按照你給的圖紙（既有幀級標註）

---

### 參數位置（若需修改路徑/輸出）
兩個檔案底部的 `__main__` 內有常數可修改：
- `SAMPLERATE`（16000）
- `PREP_PATH`（模型）
- `DS_CACHE_PATH`（HuggingFace快取）
- `DATA_PATH`（load_from_disk 的資料集路徑）
- `CSV_OUTPUT`（輸出CSV路徑）

你可以直接打開檔案改這些值再執行。

---

### 如何執行

- 自動 CTC 對齊（speechocean）：
```bash
python official/SO/speechocean_quantification.py
```

- 使用外部幀對齊（fa）：
```bash
python official/SO/fa_quantification.py
```

---

### 輸出CSV欄位（你會得到什麼）

- `speechocean_quantification.py`（時間為秒，含綜合分數）
  - `uttid, actual_phoneme, mispronounced_phoneme, start_time, end_time, confidence, prosetrior_probability, max_logit, mean_logit_margin, logit_variance, combined_score, phoneme_accuracy, mispronounced`
  - 比喻：像是攝影師給每張照片的「自動對焦時間段＋清晰度分數＋綜合評價」。

- `fa_quantification.py`（時間為幀，含不確定性指標）
  - `uttid, actual_phoneme, mispronounced_phoneme, start_frame, end_frame, posterior_prob_standard, posterior_prob_temp, max_logit, logit_margin, entropy, renyi_entropy, tsallis_entropy, mispronounced`
  - 比喻：像是手錶匠用卡尺量每個齒輪，提供「幀級位置＋多種不確定性量測」。

---

### 什麼情況用哪個？

- 用 `speechocean_quantification.py`（自動）：
  - 沒有幀級標註，想快速跑一版，並需要一個綜合分數 `combined_score`。
  - 想比較 AI 分數與人工 `phoneme_accuracy`。

- 用 `fa_quantification.py`（外部對齊）：
  - 已有高品質的強制對齊結果（幀級 `phoneme_segments`）。
  - 想研究校準/不確定性（溫度縮放、Shannon/Rényi/Tsallis 熵、多種聚合法）。

---

### 常見錯誤與排查
- 找不到資料集：確認 `DATA_PATH` 是 `datasets.load_from_disk` 可讀的資料夾。
- 字段缺失：
  - speechocean 缺 `words` 或 `phones-accuracy`：會導致人工分數為空（可以，但欄位為 None）。
  - fa 缺 `phoneme_segments` 或長度不一致：該 `uttid` 會被跳過。
- 模型無法載入：確認 `PREP_PATH` 可被 transformers 下載，或預先下載並設置 `DS_CACHE_PATH`。

---

### 簡短的操作隱喻
- speechocean：把音檔交給「智能攝影師」，他會自動框住每個音素，幫你評分並給一個整體評價。
- fa：把音檔和「精密圖紙」一起交給「手錶匠」，他會在你指定的位置做更精細的量測與不確定性分析。

需要我幫你檢查你的資料集欄位是否對應這兩個腳本的需求嗎？我可以列出應驗格式的最小範例給你對照。