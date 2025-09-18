import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def analyze_gop_metrics(csv_filepath: str):
    """
    載入GOP指標CSV檔案，計算每個指標在發音錯誤檢測上的分類效能。
    
    這個函數會遵循 Parikh et al. (2025) 的方法，為每個指標尋找
    能夠最大化MCC分數的最佳分類門檻值，然後回報在該門檻值下的各項效能指標。
    
    Args:
        csv_filepath (str): 包含GOP指標和'mispronounced'標籤的CSV檔案路徑。
    """
    try:
        df = pd.read_csv(csv_filepath)
        print(f"成功載入檔案: {csv_filepath}")
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 '{csv_filepath}'。請確認檔案名稱和路徑是否正確。")
        return

    # --- 新增：確保 'mispronounced' 欄位是布林型別 ---
    if 'mispronounced' not in df.columns:
        print("錯誤: CSV 檔案中找不到 'mispronounced' 欄位。")
        return
    df['mispronounced'] = df['mispronounced'].astype(bool)

    # --- 新增：檢查資料平衡性 ---
    print("\n'mispronounced' 標籤分布情況:")
    print(df['mispronounced'].value_counts())
    if len(df['mispronounced'].unique()) < 2:
        print("\n警告: 'mispronounced' 欄位中只存在單一類別，無法進行分類效能評估。")
        return

    # 'high'表示分數越高，錯誤發音的可能性越大
    # 'low'表示分數越低，錯誤發音的可能性越大
    metrics_to_evaluate = {
        # 基線方法
        'max_logit': 'low',
        'mean_logit_margin': 'low',
        'prosetrior_probability': 'low',
        'logit_variance': 'high',
        # 我們提出的方法
        'evt_k3': 'low',
        'skewness': 'high',
        'kurtosis': 'high',
        'autocorr_lag1': 'high',
        'entropy_mean': 'low',
        'kl_to_onehot': 'high',
        'gmm_means_0': 'low',
        'gmm_means_1': 'low',
        'gmm_vars_0': 'high',
        'gmm_vars_1': 'high',
        'gmm_weights_0': 'high',
        'gmm_weights_1': 'high'
    }
    
    results = []
    
    for metric, direction in metrics_to_evaluate.items():
        if metric not in df.columns:
            print(f"\n警告: 在資料中找不到指標 '{metric}'，跳過此項。")
            continue

        # 複製資料框以避免影響原始資料
        df_metric = df[[metric, 'mispronounced']].copy()
        
        # --- 修改：將 inf 和 -inf 取代為 NaN，以便 dropna 可以移除它們 ---
        df_metric.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 移除含有 NaN 的行
        original_count = len(df_metric)
        df_metric.dropna(inplace=True)
        
        if len(df_metric) < original_count:
            print(f"\n指標 '{metric}': 移除了 {original_count - len(df_metric)} 個含有 NaN 或 inf 的資料點。")

        if df_metric.empty:
            print(f"指標 '{metric}': 移除無效值後沒有剩餘資料，跳過。")
            continue
            
        y_true_subset = df_metric['mispronounced']
        scores = df_metric[metric]
        
        # 確保子集中仍然有兩個類別
        if len(y_true_subset.unique()) < 2:
            print(f"指標 '{metric}': 篩選後只剩下單一類別的標籤，無法計算 MCC，跳過。")
            continue

        thresholds = scores.unique()
        
        best_mcc = -1
        best_threshold = None

        for threshold in thresholds:
            y_pred = scores < threshold if direction == 'low' else scores > threshold
            mcc = matthews_corrcoef(y_true_subset, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

        if best_threshold is not None:
            final_y_pred = scores < best_threshold if direction == 'low' else scores > best_threshold
            
            # 使用 labels 參數來處理潛在的單一類別警告
            labels = [True, False]
            
            results.append({
                'Method': metric,
                'Accuracy': accuracy_score(y_true_subset, final_y_pred),
                'Precision': precision_score(y_true_subset, final_y_pred, zero_division=0, labels=labels, pos_label=True),
                'Recall': recall_score(y_true_subset, final_y_pred, zero_division=0, labels=labels, pos_label=True),
                'F1-Score': f1_score(y_true_subset, final_y_pred, zero_division=0, labels=labels, pos_label=True),
                'MCC': best_mcc
            })

    if not results:
        print("\n沒有任何指標可以成功計算效能。")
        return

    results_df = pd.DataFrame(results).sort_values(by='MCC', ascending=False)
    
    print("\n" + "="*80)
    print("       發音錯誤檢測之分類效能比較 (以 MCC 分數排序)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

# 假設您會從命令列執行，您可以保留以下部分
if __name__ == '__main__':
    # 替換成您的檔案路徑
    analyze_gop_metrics('./output/myimpl_speechocean_metrics_so5000.csv')