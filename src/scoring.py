import numpy as np
import pandas as pd

def compute_residuals(df):
    """
    Computes the residual error between the actual values and the point predictions.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'value' and 'pred_point'.
    Returns:
        pd.Series: Residuals (actual - predicted).
    """
    return df['value'] - df['pred_point']

def detect_quantile_violations(df):
    """
    Flags points where the actual value falls outside the predicted 10%-90% quantile band.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'value', 'pred_lower' (q10), and 'pred_upper' (q90).
    Returns:
        pd.Series: Binary flags (1: violation/anomaly, 0: normal).
    """
    # Flag True where value is less than lower bound or greater than upper bound
    violation = (df['value'] < df['pred_lower']) | (df['value'] > df['pred_upper'])
    
    # We only flag anomalies where we actually have predictions (exclude warmup period)
    valid_predictions = df['pred_point'].notna()
    return (violation & valid_predictions).astype(int)

def detect_zscore_violations(df, z_threshold=3.0, rolling_window=None):
    """
    Computes Z-scores on residuals and flags anomalies where the absolute Z-score
    exceeds a configurable threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'value' and 'pred_point'.
        z_threshold (float): Configurable threshold (e.g., 2.5, 3.0, 3.5).
        rolling_window (int): Size of the rolling window for running mean/std.
                             If None, computes global mean/std of residuals.
    """
    residuals = compute_residuals(df)
    
    if rolling_window:
        # Calculate running mean and std of residuals
        roll_mean = residuals.rolling(window=rolling_window, min_periods=10).mean()
        roll_std = residuals.rolling(window=rolling_window, min_periods=10).std()
        # Handle case where standard deviation is zero or NaN
        roll_std = roll_std.replace(0, np.nan).bfill().fillna(1e-6)
        
        z_scores = (residuals - roll_mean) / roll_std
    else:
        # Global z-score calculation
        mean_res = residuals.mean()
        std_res = residuals.std()
        if std_res == 0:
            std_res = 1e-6
        z_scores = (residuals - mean_res) / std_res
        
    # Exclude warmup periods
    valid_predictions = df['pred_point'].notna()
    violation = z_scores.abs() > z_threshold
    
    return (violation & valid_predictions).astype(int), z_scores

def score_anomalies(df, z_threshold=3.0, rolling_window=100):
    """
    Runs the entire anomaly scoring pipeline, adding anomaly flag columns to the DataFrame.
    
    Columns added:
      - 'residual': The raw forecasting error.
      - 'z_score': The Z-score of the residual.
      - 'anomaly_quantile': Flag based on 10%-90% prediction band violation.
      - 'anomaly_zscore': Flag based on residual Z-score exceeding z_threshold.
      - 'anomaly_combined': Flag where EITHER quantile OR Z-score triggers (high recall).
    """
    df = df.copy()
    
    # 1. Compute raw residuals
    df['residual'] = compute_residuals(df)
    
    # 2. Flag quantile band violations
    df['anomaly_quantile'] = detect_quantile_violations(df)
    
    # 3. Flag Z-score violations
    df['anomaly_zscore'], df['z_score'] = detect_zscore_violations(
        df, z_threshold=z_threshold, rolling_window=rolling_window
    )
    
    # 4. Combined flag (EITHER triggers an anomaly)
    df['anomaly_combined'] = ((df['anomaly_quantile'] == 1) | (df['anomaly_zscore'] == 1)).astype(int)
    
    return df
