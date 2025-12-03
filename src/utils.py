"""
Utility functions for data processing, market day alignment, and reproducibility.
"""

import pandas as pd
import numpy as np
import torch
import random
from datetime import datetime, time
from typing import Optional, Tuple
import pytz


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def align_to_market_day(timestamp: pd.Timestamp, tz: str = 'US/Eastern') -> pd.Timestamp:
    """
    Align a timestamp to its corresponding market day.
    
    News/posts after 16:00 ET are assigned to the next trading day to prevent leakage.
    
    Args:
        timestamp: Input timestamp
        tz: Timezone string (default: US/Eastern)
    
    Returns:
        Market-aligned date
    """
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC')
    
    # Convert to Eastern Time
    et_time = timestamp.tz_convert(tz)
    
    # If after 4 PM, assign to next day
    if et_time.time() > time(16, 0):
        et_time = et_time + pd.Timedelta(days=1)
    
    return et_time.normalize()


def get_business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get business days between two dates.
    
    Args:
        start_date: Start date string
        end_date: End date string
    
    Returns:
        DatetimeIndex of business days
    """
    return pd.bdate_range(start=start_date, end=end_date)


def create_sequences(data: np.ndarray, lookback: int, target: Optional[np.ndarray] = None) -> Tuple:
    """
    Create sequences for LSTM training.
    
    Args:
        data: Feature array (n_samples, n_features)
        lookback: Number of timesteps to look back
        target: Target array (optional)
    
    Returns:
        Tuple of (sequences, targets) or just sequences if no target
    """
    X = []
    y = [] if target is not None else None
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        if target is not None:
            y.append(target[i])
    
    X = np.array(X)
    
    if target is not None:
        y = np.array(y)
        return X, y
    
    return X


def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage returns."""
    return prices.pct_change(periods=periods)


def calculate_volatility(returns: pd.Series, window: int = 5) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        window: RSI window (default: 14)
    
    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'signal': signal_line,
        'histogram': histogram
    })


def split_by_date(df: pd.DataFrame, 
                  train_end: str = '2021-12-31',
                  val_end: str = '2023-12-31') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by date for time series.
    
    Args:
        df: DataFrame with DatetimeIndex
        train_end: End date for training set
        val_end: End date for validation set
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[df.index > val_end]
    
    return train, val, test


def normalize_features(train: np.ndarray, 
                       val: np.ndarray, 
                       test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Normalize features using training set statistics.
    
    Args:
        train: Training features
        val: Validation features
        test: Test features
    
    Returns:
        Tuple of (normalized_train, normalized_val, normalized_test, scaler_params)
    """
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1  # Avoid division by zero
    
    train_norm = (train - mean) / std
    val_norm = (val - mean) / std
    test_norm = (test - mean) / std
    
    scaler_params = {'mean': mean, 'std': std}
    
    return train_norm, val_norm, test_norm, scaler_params


def inverse_transform(data: np.ndarray, scaler_params: dict) -> np.ndarray:
    """Inverse transform normalized data."""
    return data * scaler_params['std'] + scaler_params['mean']


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' - whether lower or higher is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Direction accuracy
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)
    direction_acc = np.mean(true_direction == pred_direction)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_acc
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty print metrics."""
    print(f"\n{prefix} Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Direction Accuracy: {metrics['direction_accuracy']:.2%}")
