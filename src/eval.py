from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def calculate_metrics(y_true, y_pred, y_score=None):
    """
    Calculates classification metrics for anomaly detection: Precision, Recall, F1, and AUC-ROC.
    
    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted binary anomaly labels (0 or 1).
        y_score (array-like): Continuous anomaly scores (used to calculate AUC-ROC).
    Returns:
        dict: Dictionary containing metric scores.
    """
    # Point-wise metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = np.nan
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            # Can fail if there's only one class present in y_true
            pass
            
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

import numpy as np

def evaluate_all_methods(scored_df, label_col='label', timesfm_cols=['anomaly_quantile', 'anomaly_zscore', 'anomaly_combined'], iforest_col='anomaly_iforest', iforest_score='score_iforest', lstm_col='anomaly_lstm', lstm_score='score_lstm'):
    """
    Computes and aggregates evaluation metrics for TimesFM (quantile, z-score, combined),
    Isolation Forest, and LSTM Autoencoder.
    
    Args:
        scored_df (pd.DataFrame): DataFrame containing predictions from all models.
    """
    # Exclude warmup indices (where predictions are NaN)
    eval_df = scored_df.dropna(subset=['pred_point']).copy()
    
    y_true = eval_df[label_col].values
    results = {}
    
    # 1. Evaluate TimesFM methods
    # For TimesFM, we can use the absolute residual or z-score as the continuous anomaly score
    res_score = eval_df['residual'].abs().values
    z_score_abs = eval_df['z_score'].abs().values
    
    results['TimesFM (Quantile)'] = calculate_metrics(y_true, eval_df[timesfm_cols[0]].values, res_score)
    results['TimesFM (Z-Score)'] = calculate_metrics(y_true, eval_df[timesfm_cols[1]].values, z_score_abs)
    results['TimesFM (Combined)'] = calculate_metrics(y_true, eval_df[timesfm_cols[2]].values, z_score_abs)
    
    # 2. Evaluate Isolation Forest
    if iforest_col in eval_df.columns:
        iforest_scores = eval_df[iforest_score].values if iforest_score in eval_df.columns else None
        results['Isolation Forest'] = calculate_metrics(y_true, eval_df[iforest_col].values, iforest_scores)
        
    # 3. Evaluate LSTM Autoencoder
    if lstm_col in eval_df.columns:
        lstm_scores = eval_df[lstm_score].values if lstm_score in eval_df.columns else None
        results['LSTM Autoencoder'] = calculate_metrics(y_true, eval_df[lstm_col].values, lstm_scores)
        
    # Format as DataFrame
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "Method"
    
    return metrics_df
