"""
Feature engineering and preprocessing utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Stock price data with OHLCV columns
    
    Returns:
    --------
    pd.DataFrame
        Data with additional technical indicator columns
    """
    data = df.copy()
    
    # Returns
    data['return_1d'] = data['Close'].pct_change()
    data['return_5d'] = data['Close'].pct_change(5)
    
    # Volatility
    data['volatility_5d'] = data['return_1d'].rolling(5).std()
    data['volatility_10d'] = data['return_1d'].rolling(10).std()
    
    # Overnight return
    data['overnight_return'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # Volume ratio
    data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
    
    # Moving averages
    data['ma5'] = data['Close'].rolling(5).mean()
    data['ma20'] = data['Close'].rolling(20).mean()
    data['price_to_ma5'] = data['Close'] / data['ma5']
    data['price_to_ma20'] = data['Close'] / data['ma20']
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_histogram'] = data['macd'] - data['macd_signal']
    
    return data

def add_sentiment_features(price_data, sentiment_data):
    """
    Add sentiment features to price data
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        Stock price data
    sentiment_data : pd.DataFrame
        Sentiment scores by date
    
    Returns:
    --------
    pd.DataFrame
        Combined data with sentiment features
    """
    if sentiment_data is None or len(sentiment_data) == 0:
        # Add dummy sentiment features
        price_data['pos'] = 0.0
        price_data['neg'] = 0.0
        price_data['neu'] = 1.0
        price_data['compound'] = 0.0
        price_data['sent_count'] = 0.0
        price_data['pos_surprise'] = 0.0
        price_data['neg_surprise'] = 0.0
        price_data['compound_surprise'] = 0.0
        return price_data
    
    # Merge sentiment data
    merged = price_data.merge(
        sentiment_data,
        left_index=True,
        right_on='date',
        how='left'
    )
    
    # Fill missing sentiment with neutral
    sentiment_cols = ['pos', 'neg', 'neu', 'compound', 'sent_count',
                     'pos_surprise', 'neg_surprise', 'compound_surprise']
    for col in sentiment_cols:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)
    
    return merged

def prepare_features(data, feature_list=None):
    """
    Prepare features for model input
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw stock data
    feature_list : list
        List of feature names to use
    
    Returns:
    --------
    np.ndarray
        Scaled feature array
    """
    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    
    # Default feature list
    if feature_list is None:
        feature_list = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'return_1d', 'return_5d', 'volatility_5d', 'volatility_10d',
            'overnight_return', 'volume_ratio', 'price_to_ma5', 'price_to_ma20',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'pos', 'neg', 'neu', 'compound', 'sent_count',
            'pos_surprise', 'neg_surprise', 'compound_surprise'
        ]
    
    # Add dummy sentiment if not present
    if 'pos' not in data.columns:
        data = add_sentiment_features(data, None)
    
    # Select features
    features = data[feature_list].copy()
    
    # Drop rows with NaN
    features = features.dropna()
    
    return features

def create_sequences(data, lookback=30):
    """
    Create sequences for LSTM input
    
    Parameters:
    -----------
    data : np.ndarray
        Feature array
    lookback : int
        Number of time steps to look back
    
    Returns:
    --------
    np.ndarray
        3D array of shape (samples, lookback, features)
    """
    sequences = []
    for i in range(lookback, len(data)):
        sequences.append(data[i-lookback:i])
    return np.array(sequences)
