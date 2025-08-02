# Official 程式碼執行流程指南

本文件提供執行 official 目錄中原始碼的完整流程說明。

## 📋 目錄結構概覽

```
official/
├── README.md                    # 專案說明文件
├── requirements.txt             # Python 依賴套件
├── MPC/                        # MPC 資料集處理
│   ├── mpc_ctc_segment_quantification.py
│   └── mpc_quantification_2.py
├── MPC_eval/                   # MPC 評估
│   └── mpc_evaluate.ipynb
├── SO/                         # SpeechOcean 資料集處理
│   ├── fa_quantification.py
│   └── speechocean_quantification.py
└── SO_eval/                    # SpeechOcean 評估
    └── so_evaluation.ipynb
```

## 🚀 執行前準備

### 1. 環境設定

#### 1.1 安裝 Python 依賴套件

```bash
# 進入 official 目錄
cd official

# 安裝依賴套件
pip install -r requirements.txt
```

**注意事項：**
- 需要 Python 3.8+ 版本
- 建議使用虛擬環境
- 需要 CUDA 支援的 GPU（推薦）或 CPU

#### 1.2 主要依賴套件說明

- `torch==2.6.0` 或 `torch==2.4.1`: PyTorch 深度學習框架
- `transformers`: Hugging Face Transformers 庫
- `ctc_segmentation==1.7.4`: CTC 分割工具
- `datasets==3.6.0`: Hugging Face 資料集庫
- `numpy==2.3.1`, `pandas==2.3.1`: 資料處理庫

### 2. 資料準備

#### 2.1 資料集路徑設定

程式碼中的資料路徑需要根據您的環境進行調整：

**MPC 資料集：**
```python
DATA_PATH = "/vol/tensusers6/aparikh/PhD/data/mpc/simulated_error_mpc"
```

**SpeechOcean 資料集：**
```python
DATA_PATH = "/vol/tensusers6/aparikh/PhD/data/spo762/so_everything_cmu_ipa"
```

**快取目錄：**
```python
DS_CACHE_PATH = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/cache_dir"
```

#### 2.2 輸出路徑設定

**MPC 輸出：**
```python
CSV_OUTPUT = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/quantification/mpc_evaluation/mpc_CTC_SEGMENT_logits_max.csv"
```

**SpeechOcean 輸出：**
```python
CSV_OUTPUT = "/vol/tensusers6/aparikh/PhD/CTC-based-GOP/quantification/speechocean_evaluation/phoneme_alignment_CTC_SEGMENT_mean.csv"
```

## 🔧 執行流程

### 階段 1: 資料量化處理

#### 1.1 MPC 資料集處理

**執行檔案：** `MPC/mpc_ctc_segment_quantification.py`

**功能：**
- 使用 Wav2Vec2 模型進行音素對齊
- 計算 GOP 分數和相關指標
- 生成包含量化結果的 CSV 檔案

**執行命令：**
```bash
cd official/MPC
python mpc_ctc_segment_quantification.py
```

**主要輸出指標：**
- `prosetrior_probability`: 後驗機率
- `max_logit`: 最大 logit 值
- `mean_logit_margin`: 平均 logit 邊距
- `logit_variance`: logit 變異數
- `combined_score`: 組合分數

#### 1.2 SpeechOcean 資料集處理

**執行檔案：** `SO/speechocean_quantification.py`

**功能：**
- 處理 SpeechOcean762 資料集
- 提取音素準確度資訊
- 生成量化結果 CSV 檔案

**執行命令：**
```bash
cd official/SO
python speechocean_quantification.py
```

**主要輸出指標：**
- 與 MPC 相同的量化指標
- `phoneme_accuracy`: 人工評分音素準確度

### 階段 2: 模型評估

#### 2.1 MPC 評估

**執行檔案：** `MPC_eval/mpc_evaluate.ipynb`

**功能：**
- 載入 MPC 量化結果
- 計算各種評估指標
- 生成 ROC 曲線和混淆矩陣
- 尋找最佳閾值

**執行方式：**
```bash
cd official/MPC_eval
jupyter notebook mpc_evaluate.ipynb
```

**評估指標：**
- Accuracy, Precision, Recall, F1 Score
- Matthews Correlation Coefficient (MCC)
- ROC AUC Score
- 最佳閾值分析

#### 2.2 SpeechOcean 評估

**執行檔案：** `SO_eval/so_evaluation.ipynb`

**功能：**
- 載入 SpeechOcean 量化結果
- 進行詳細的統計分析
- 比較不同評分方法
- 生成視覺化圖表

**執行方式：**
```bash
cd official/SO_eval
jupyter notebook so_evaluation.ipynb
```

## ⚙️ 配置參數說明

### 模型配置

```python
PREP_PATH = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"  # 預訓練模型
SAMPLERATE = 16000  # 音頻採樣率
```

### 計算參數

```python
alpha = 0.7  # 組合分數權重參數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 計算設備
```

## 📊 輸出檔案說明

### CSV 檔案格式

**MPC 輸出欄位：**
- `uttid`: 語音檔案 ID
- `actual_phoneme`: 實際音素
- `mispronounced_phoneme`: 錯誤發音音素
- `start_time`, `end_time`: 時間戳記
- `confidence`: 信心度
- `prosetrior_probability`: 後驗機率
- `max_logit`: 最大 logit 值
- `mean_logit_margin`: 平均 logit 邊距
- `logit_variance`: logit 變異數
- `combined_score`: 組合分數
- `mispronounced`: 是否為錯誤發音

**SpeechOcean 額外欄位：**
- `phoneme_accuracy`: 人工評分音素準確度

## 🔍 故障排除

### 常見問題

1. **CUDA 記憶體不足**
   - 減少批次大小
   - 使用 CPU 模式
   - 清理 GPU 記憶體

2. **資料路徑錯誤**
   - 檢查資料集路徑是否正確
   - 確認檔案權限
   - 驗證資料格式

3. **依賴套件版本衝突**
   - 使用虛擬環境
   - 檢查 requirements.txt 版本
   - 更新或降級衝突套件

### 效能優化建議

1. **GPU 加速**
   - 確保 CUDA 版本與 PyTorch 相容
   - 使用適當的批次大小
   - 監控 GPU 記憶體使用

2. **記憶體管理**
   - 定期清理變數
   - 使用生成器處理大型資料集
   - 分批處理資料

## 📈 結果解讀

### 評估指標說明

- **GOP Score**: 傳統的 Goodness of Pronunciation 分數
- **Logit-based Metrics**: 基於神經網路 logit 的新指標
- **Combined Score**: 結合多種指標的綜合評分

### 最佳實踐

1. **閾值選擇**: 使用 MCC 最大化作為主要指標
2. **模型比較**: 比較不同評分方法的效能
3. **統計分析**: 進行顯著性檢定
4. **視覺化**: 使用 ROC 曲線和混淆矩陣

## 📝 注意事項

1. **資料隱私**: 確保遵守資料使用協議
2. **計算資源**: 大型資料集需要充足的計算資源
3. **結果驗證**: 建議使用交叉驗證
4. **文檔記錄**: 記錄所有實驗參數和結果

## 🔗 相關資源

- [論文連結](https://arxiv.org/abs/2506.12067)
- [Interspeech 2025](https://www.interspeech2025.org)
- [Wav2Vec2 模型](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
- [CTC Segmentation](https://github.com/lumaku/ctc-segmentation) 