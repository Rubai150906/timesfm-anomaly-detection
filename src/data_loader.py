import os
import urllib.request
import json
import pandas as pd
import numpy as np

# NAB URLs
BASE_URL = "https://raw.githubusercontent.com/numenta/NAB/master"
DATA_URLS = {
    "temperature": f"{BASE_URL}/data/realKnownCause/ambient_temperature_system_failure.csv",
    "cpu": f"{BASE_URL}/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv"
}
LABELS_URL = f"{BASE_URL}/labels/combined_windows.json"

def download_nab_data(dest_dir="data"):
    """
    Downloads the selected NAB CSV files and combined_windows.json labels
    and saves them in the dest_dir.
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    # Download CSVs
    downloaded_paths = {}
    for name, url in DATA_URLS.items():
        filename = os.path.basename(url)
        path = os.path.join(dest_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename} from NAB repo...")
            urllib.request.urlretrieve(url, path)
            print(f"Saved to {path}")
        else:
            print(f"{filename} already exists at {path}")
        downloaded_paths[name] = path
        
    # Download Labels JSON
    labels_path = os.path.join(dest_dir, "combined_windows.json")
    if not os.path.exists(labels_path):
        print("Downloading combined_windows.json from NAB repo...")
        urllib.request.urlretrieve(LABELS_URL, labels_path)
        print(f"Saved to {labels_path}")
    else:
        print(f"combined_windows.json already exists at {labels_path}")
        
    downloaded_paths["labels"] = labels_path
    return downloaded_paths

def load_series_with_labels(csv_path, labels_json_path, repo_relative_path):
    """
    Loads a time series CSV, parses timestamps, and maps ground-truth anomaly windows
    from the combined_windows.json file to binary labels (0: normal, 1: anomaly).
    
    Args:
        csv_path (str): Local path to the downloaded CSV file.
        labels_json_path (str): Local path to combined_windows.json.
        repo_relative_path (str): The relative path in the NAB repo (e.g., 'realKnownCause/ambient_temperature_system_failure.csv')
                                  used to look up the windows in the JSON labels file.
    """
    # 1. Load CSV
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 2. Load Labels
    with open(labels_json_path, "r") as f:
        labels_dict = json.load(f)
        
    anomaly_windows = labels_dict.get(repo_relative_path, [])
    
    # 3. Create labels column (1 if timestamp falls inside any anomaly window, 0 otherwise)
    df["label"] = 0
    for start_str, end_str in anomaly_windows:
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        # Match timestamps in window
        df.loc[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt), "label"] = 1
        
    print(f"Loaded {repo_relative_path} with {len(df)} rows.")
    print(f"Found {df['label'].sum()} anomalous timestamps out of {len(df)} total ({df['label'].mean():.2%}).")
    return df

def preprocess_series(df, freq=None):
    """
    Checks for missing timestamps, resamples if requested, interpolates missing values,
    and returns a clean DataFrame with a DatetimeIndex.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing 'timestamp', 'value', and 'label'.
        freq (str): Optional pandas resampling frequency (e.g., '1H' or '5T').
    """
    df = df.copy()
    df.set_index("timestamp", inplace=True)
    
    # If resampling is requested to fill in gaps or standardize
    if freq:
        # Resample index, taking the mean of values, and max of labels (to preserve anomaly status)
        resampled_val = df["value"].resample(freq).mean()
        resampled_lbl = df["label"].resample(freq).max().fillna(0).astype(int)
        
        df = pd.DataFrame({
            "value": resampled_val,
            "label": resampled_lbl
        })
    
    # Check for NaN and interpolate values
    nan_count = df["value"].isna().sum()
    if nan_count > 0:
        print(f"Found {nan_count} missing value(s). Filling with linear interpolation...")
        df["value"] = df["value"].interpolate(method="linear")
        # Forward fill labels if any NaN (though resample fills with 0 by default)
        df["label"] = df["label"].fillna(0).astype(int)
        
    df.reset_index(inplace=True)
    return df

def prepare_splits(df, context_len):
    """
    Since we are using TimesFM in zero-shot mode, we do not perform traditional model training.
    However, we need a baseline context to start predicting.
    This function splits the data into:
    1. A warmup context (first context_len elements) to bootstrap the rolling forecast.
    2. An evaluation target set (remaining elements) where forecasts are generated and anomalies evaluated.
    """
    if len(df) <= context_len:
        raise ValueError(f"Time series length ({len(df)}) must be greater than context_len ({context_len})")
        
    warmup_df = df.iloc[:context_len].reset_index(drop=True)
    eval_df = df.iloc[context_len:].reset_index(drop=True)
    
    print(f"Warmup context split: {len(warmup_df)} timestamps.")
    print(f"Evaluation target split: {len(eval_df)} timestamps.")
    return warmup_df, eval_df
