# myimpl/analyze_metrics_fast.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import time

def analyze_gop_metrics(csv_filepath: str, max_thresholds: int = 200):
    """
    載入GOP指標CSV檔案，計算每個指標在發音錯誤檢測上的分類效能。

    優化版本：使用較少的閾值來加速計算。
    """
    start_time = time.time()

    try:
        print(f"載入檔案: {csv_filepath}")
        df = pd.read_csv(csv_filepath)
        print(f"成功載入檔案，資料形狀: {df.shape}")
    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 '{csv_filepath}'。")
        return

    # 確保 'mispronounced' 欄位是布林型別
    if 'mispronounced' not in df.columns:
        print("錯誤: CSV 檔案中找不到 'mispronounced' 欄位。")
        return
    df['mispronounced'] = df['mispronounced'].astype(bool)

    # 檢查資料平衡性
    print("\n'mispronounced' 標籤分布情況:")
    print(df['mispronounced'].value_counts())
    if len(df['mispronounced'].unique()) < 2:
        print("\n警告: 'mispronounced' 欄位中只存在單一類別，無法進行分類效能評估。")
        return

    # 待評估的指標
    metrics_to_evaluate = {
        'max_logit': 'low',
        'mean_logit_margin': 'low',
        'prosetrior_probability': 'low',
        'logit_variance': 'high',
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
    total_metrics = len([m for m in metrics_to_evaluate.keys() if m in df.columns])
    processed_metrics = 0

    for metric, direction in metrics_to_evaluate.items():
        if metric not in df.columns:
            print(f"\n警告: 在資料中找不到指標 '{metric}'，跳過此項。")
            continue

        processed_metrics += 1
        metric_start_time = time.time()
        print(f"\n[{processed_metrics}/{total_metrics}] 處理指標: {metric}")

        # 複製並清理資料
        df_metric = df[[metric, 'mispronounced']].copy()
        df_metric.replace([np.inf, -np.inf], np.nan, inplace=True)

        original_count = len(df_metric)
        df_metric.dropna(inplace=True)

        if len(df_metric) < original_count:
            print(f"  移除了 {original_count - len(df_metric)} 個含有 NaN 或 inf 的資料點")

        if df_metric.empty:
            print(f"  移除無效值後沒有剩餘資料，跳過")
            continue

        y_true_subset = df_metric['mispronounced']
        scores = df_metric[metric]

        # 確保有兩個類別
        if len(y_true_subset.unique()) < 2:
            print(f"  篩選後只剩下單一類別的標籤，跳過")
            continue

        # 優化閾值選擇：使用分位數
        print(f"  原始唯一值數量: {scores.nunique()}")
        percentiles = np.linspace(1, 99, max_thresholds)  # 避免極值
        thresholds = np.percentile(scores, percentiles)
        thresholds = np.unique(thresholds)

        print(f"  使用 {len(thresholds)} 個閾值進行搜尋")

        best_mcc = -1
        best_threshold = None

        # 閾值搜尋
        for threshold in thresholds:
            y_pred = scores < threshold if direction == 'low' else scores > threshold

            # 確保預測有兩個類別
            if len(np.unique(y_pred)) < 2:
                continue

            try:
                mcc = matthews_corrcoef(y_true_subset, y_pred)
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_threshold = threshold
            except:
                continue

        if best_threshold is not None:
            final_y_pred = scores < best_threshold if direction == 'low' else scores > best_threshold

            try:
                results.append({
                    'Method': metric,
                    'Accuracy': accuracy_score(y_true_subset, final_y_pred),
                    'Precision': precision_score(y_true_subset, final_y_pred, zero_division=0),
                    'Recall': recall_score(y_true_subset, final_y_pred, zero_division=0),
                    'F1-Score': f1_score(y_true_subset, final_y_pred, zero_division=0),
                    'MCC': best_mcc
                })

                metric_time = time.time() - metric_start_time
                print(f"  完成! 最佳 MCC: {best_mcc:.4f}, 耗時: {metric_time:.1f}s")
            except Exception as e:
                print(f"  計算指標時發生錯誤: {e}")
        else:
            print(f"  無法找到有效的閾值")

    if not results:
        print("\n沒有任何指標可以成功計算效能。")
        return

    # 顯示結果
    results_df = pd.DataFrame(results).sort_values(by='MCC', ascending=False)

    total_time = time.time() - start_time
    print(f"\n總耗時: {total_time:.1f}s")
    print("\n" + "="*80)
    print("       發音錯誤檢測之分類效能比較 (以 MCC 分數排序)")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

    return results_df

if __name__ == '__main__':
    analyze_gop_metrics('output/myimpl_speechocean_metrics_so5000_v2.csv')