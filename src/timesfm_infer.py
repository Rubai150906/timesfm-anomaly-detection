import torch
import numpy as np
import pandas as pd
import timesfm

def load_timesfm_model(checkpoint_path="google/timesfm-2.5-200m-pytorch", max_context=1024, max_horizon=256):
    """
    Initializes and compiles the TimesFM 2.5 PyTorch model.
    """
    print(f"Initializing TimesFM 2.5 from {checkpoint_path}...")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(checkpoint_path)
    
    print("Compiling model configuration...")
    model.compile(
        timesfm.ForecastConfig(
            max_context=max_context,
            max_horizon=max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
        )
    )
    print("Model compiled successfully and ready for zero-shot forecasting.")
    return model

def rolling_forecast(model, df, context_len=512, horizon=32, step_size=None):
    """
    Performs a rolling-window zero-shot forecast over the time series.
    
    Args:
        model: Compiled TimesFM model.
        df (pd.DataFrame): Time series DataFrame with columns ['timestamp', 'value'].
        context_len (int): Length of historical context passed to the model.
                           Tradeoff: Longer context (up to 16k for v2.5) gives the model
                           more history to learn trends/seasonal patterns, but increases memory usage.
                           512 is a standard robust choice for daily/hourly series.
        horizon (int): Number of steps to forecast in the future.
                       Tradeoff: Shorter horizon (e.g. 1 or 8) is more accurate because we forecast
                       near-future steps, but requires more model calls. Longer horizon (e.g. 32 or 64)
                       runs significantly faster but predictions further out will drift.
        step_size (int): Slide step size. Defaults to `horizon` (block-rolling), which is highly
                         optimized and fast. If set to 1, does fully sliding point-by-point rolling
                         (computationally heavy).
                         
    Returns:
        pd.DataFrame: Original DataFrame extended with forecasting columns:
                      'pred_point', 'pred_lower' (q10), 'pred_median' (q50), 'pred_upper' (q90)
                      Note: The first `context_len` steps will have NaN predictions.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    
    # Initialize output arrays with NaN
    pred_points = np.full(n, np.nan)
    pred_lowers = np.full(n, np.nan)
    pred_medians = np.full(n, np.nan)
    pred_uppers = np.full(n, np.nan)
    
    if step_size is None:
        step_size = horizon
        
    print(f"Starting rolling forecast. Series length: {n}, Context: {context_len}, Horizon: {horizon}, Step Size: {step_size}")
    
    # Iterate through the series starting from context_len
    i = context_len
    calls = 0
    
    # To batch predictions and speed it up, we can loop
    while i < n:
        # Context window: historical observations
        # If we have more than max_context observations, clip to the last context_len
        start_idx = max(0, i - context_len)
        context = df['value'].iloc[start_idx:i].values
        
        # TimesFM expects a list of 1-D arrays
        # The forecast call returns: point_forecast, quantile_forecast
        point_f, quant_f = model.forecast(
            horizon=horizon,
            inputs=[context]
        )
        
        # Determine actual number of steps to fill in this block
        steps_to_fill = min(horizon, n - i)
        
        # Quantile index mapping (based on TimesFM 2.5 outputs of shape [batch, horizon, 10]):
        # Index 0: Mean, Index 1: q10, Index 5: q50 (Median), Index 9: q90
        pred_points[i:i+steps_to_fill] = point_f[0][:steps_to_fill]
        pred_lowers[i:i+steps_to_fill] = quant_f[0][:steps_to_fill, 1]
        pred_medians[i:i+steps_to_fill] = quant_f[0][:steps_to_fill, 5]
        pred_uppers[i:i+steps_to_fill] = quant_f[0][:steps_to_fill, 9]
        
        # Advance rolling window
        i += step_size
        calls += 1
        
        if calls % 20 == 0 or i >= n:
            pct = min(100.0, (i / n) * 100)
            print(f"  Processed {i}/{n} steps ({pct:.1f}%) - {calls} model calls.")
            
    df['pred_point'] = pred_points
    df['pred_lower'] = pred_lowers
    df['pred_median'] = pred_medians
    df['pred_upper'] = pred_uppers
    
    return df
