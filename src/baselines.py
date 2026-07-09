import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ========================================== #
# 1. Isolation Forest Baseline               #
# ========================================== #

def engineer_features(df, window_sizes=[6, 12, 24], lags=[1, 2, 3]):
    """
    Engineers temporal features (lags, rolling mean, rolling std) for tabular models.
    """
    df_feat = df.copy()
    features = []
    
    # Value lags
    for lag in lags:
        col_name = f'lag_{lag}'
        df_feat[col_name] = df_feat['value'].shift(lag)
        features.append(col_name)
        
    # Rolling statistics
    for w in window_sizes:
        mean_col = f'roll_mean_{w}'
        std_col = f'roll_std_{w}'
        df_feat[mean_col] = df_feat['value'].rolling(window=w, min_periods=1).mean()
        df_feat[std_col] = df_feat['value'].rolling(window=w, min_periods=1).std().fillna(0)
        features.extend([mean_col, std_col])
        
    # Difference (derivative)
    df_feat['diff_1'] = df_feat['value'].diff().fillna(0)
    features.append('diff_1')
    
    # Fill remaining NaNs from early lags
    df_feat = df_feat.fillna(method='bfill').fillna(0)
    
    return df_feat, features

def run_isolation_forest(train_df, test_df, contamination=0.05):
    """
    Trains an Isolation Forest on the training (warmup) set and predicts anomalies on the test set.
    """
    # 1. Engineer features
    # Concat train and test to engineer features consistently, then split back
    split_idx = len(train_df)
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    full_df, features = engineer_features(full_df)
    
    X_train = full_df[features].iloc[:split_idx].values
    X_test = full_df[features].iloc[split_idx:].values
    
    # 2. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Fit model
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    model.fit(X_train_scaled)
    
    # 4. Predict
    # IsolationForest returns -1 for anomalies, 1 for normal
    test_preds = model.predict(X_test_scaled)
    anomaly_flags = (test_preds == -1).astype(int)
    
    # Score represents anomaly score (negative of decision function)
    anomaly_scores = -model.decision_function(X_test_scaled)
    
    result_df = test_df.copy()
    result_df['anomaly_iforest'] = anomaly_flags
    result_df['score_iforest'] = anomaly_scores
    
    return result_df

# ========================================== #
# 2. LSTM Autoencoder Baseline               #
# ========================================== #

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, input_dim=1, hidden_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_linear = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        
        # Encode: get final hidden state
        _, (hidden, _) = self.encoder_lstm(x)
        # hidden shape: (1, batch, hidden_dim)
        
        # Repeat hidden state seq_len times
        hidden = hidden.transpose(0, 1) # (batch, 1, hidden_dim)
        repeated_hidden = hidden.repeat(1, self.seq_len, 1) # (batch, seq_len, hidden_dim)
        
        # Decode
        decoder_out, _ = self.decoder_lstm(repeated_hidden)
        reconstruction = self.output_linear(decoder_out)
        
        return reconstruction

def create_sequences(data, seq_len):
    """
    Creates overlapping sequences for LSTM input.
    """
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

def run_lstm_autoencoder(train_df, test_df, seq_len=24, hidden_dim=16, epochs=15, batch_size=32, lr=0.002, threshold_pct=95):
    """
    Trains an LSTM Autoencoder on normal warmup training data, calculates reconstruction
    error on the evaluation test set, and flags anomalies.
    """
    # 1. Scale data based on training set
    scaler = StandardScaler()
    train_vals = scaler.fit_transform(train_df['value'].values.reshape(-1, 1))
    test_vals = scaler.transform(test_df['value'].values.reshape(-1, 1))
    
    # 2. Create sequences
    X_train = create_sequences(train_vals, seq_len) # shape: (num_train_seq, seq_len, 1)
    # For test, we pad the start with the end of train data so we get a sequence for every test point
    padded_test_vals = np.concatenate([train_vals[-seq_len+1:], test_vals], axis=0)
    X_test = create_sequences(padded_test_vals, seq_len) # shape: (num_test_seq, seq_len, 1)
    
    # Convert to PyTorch Tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model instantiation
    model = LSTMAutoencoder(seq_len=seq_len, input_dim=1, hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training loop
    model.train()
    print(f"Training LSTM Autoencoder on {device} ({epochs} epochs)...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
            
        epoch_loss /= len(train_dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.5f}")
            
    # 5. Evaluate training errors to set reconstruction threshold
    model.eval()
    train_losses = []
    with torch.no_grad():
        for batch in train_loader:
            x_batch = batch[0].to(device)
            reconstructed = model(x_batch)
            # Calculate MSE per sequence in batch
            losses = torch.mean((reconstructed - x_batch) ** 2, dim=[1, 2]).cpu().numpy()
            train_losses.extend(losses)
            
    threshold = np.percentile(train_losses, threshold_pct)
    print(f"LSTM Autoencoder reconstruction error threshold (P{threshold_pct}): {threshold:.5f}")
    
    # 6. Predict on test set
    test_losses = []
    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # process test sequences
        reconstructed_test = model(test_tensor)
        # MSE per test sequence
        test_losses = torch.mean((reconstructed_test - test_tensor) ** 2, dim=[1, 2]).cpu().numpy()
        
    # Anomaly flags: 1 if test reconstruction error exceeds threshold, 0 otherwise
    anomaly_flags = (test_losses > threshold).astype(int)
    
    result_df = test_df.copy()
    result_df['anomaly_lstm'] = anomaly_flags
    result_df['score_lstm'] = test_losses
    
    return result_df
