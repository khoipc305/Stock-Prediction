"""
Create supervised learning dataset by merging prices with sentiment and building features.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .utils import (
    calculate_returns, 
    calculate_volatility, 
    calculate_rsi, 
    calculate_macd,
    split_by_date
)


def load_prices(path: str) -> pd.DataFrame:
    """Load price data."""
    print(f"Loading prices from {path}...")
    df = pd.read_parquet(path)
    df['Date'] = pd.to_datetime(df['Date'])
    # Ensure timezone-naive for consistent merging
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_localize(None)
    return df


def load_sentiment(path: str) -> pd.DataFrame:
    """Load sentiment data."""
    print(f"Loading sentiment from {path}...")
    df = pd.read_parquet(path)
    df['market_day'] = pd.to_datetime(df['market_day'])
    # Remove timezone to match price data
    if df['market_day'].dt.tz is not None:
        df['market_day'] = df['market_day'].dt.tz_localize(None)
    return df


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build technical indicators for each ticker.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with technical features added
    """
    print("Building technical features...")
    
    features = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date')
        
        # Returns
        ticker_df['return_1d'] = calculate_returns(ticker_df['Close'], periods=1)
        ticker_df['return_5d'] = calculate_returns(ticker_df['Close'], periods=5)
        
        # Volatility
        ticker_df['volatility_5d'] = calculate_volatility(ticker_df['return_1d'], window=5)
        ticker_df['volatility_10d'] = calculate_volatility(ticker_df['return_1d'], window=10)
        
        # Overnight return
        ticker_df['overnight_return'] = (ticker_df['Open'] - ticker_df['Close'].shift(1)) / ticker_df['Close'].shift(1)
        
        # Volume features
        ticker_df['volume_ma5'] = ticker_df['Volume'].rolling(5).mean()
        ticker_df['volume_ratio'] = ticker_df['Volume'] / ticker_df['volume_ma5']
        
        # Price momentum
        ticker_df['price_ma5'] = ticker_df['Close'].rolling(5).mean()
        ticker_df['price_ma20'] = ticker_df['Close'].rolling(20).mean()
        ticker_df['price_to_ma5'] = ticker_df['Close'] / ticker_df['price_ma5']
        ticker_df['price_to_ma20'] = ticker_df['Close'] / ticker_df['price_ma20']
        
        # RSI
        ticker_df['rsi'] = calculate_rsi(ticker_df['Close'], window=14)
        
        # MACD
        macd_df = calculate_macd(ticker_df['Close'])
        ticker_df['macd'] = macd_df['macd']
        ticker_df['macd_signal'] = macd_df['signal']
        ticker_df['macd_histogram'] = macd_df['histogram']
        
        features.append(ticker_df)
    
    result = pd.concat(features, ignore_index=True)
    print(f"Built technical features for {len(result)} rows")
    
    return result


def merge_with_sentiment(price_df: pd.DataFrame, 
                        sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price data with sentiment features.
    
    Args:
        price_df: DataFrame with price and technical features
        sentiment_df: DataFrame with daily sentiment
    
    Returns:
        Merged DataFrame
    """
    print("Merging prices with sentiment...")
    
    # Merge on ticker and date
    merged = price_df.merge(
        sentiment_df,
        left_on=['ticker', 'Date'],
        right_on=['ticker', 'market_day'],
        how='left'
    )
    
    # Drop duplicate date column
    if 'market_day' in merged.columns:
        merged = merged.drop(columns=['market_day'])
    
    # Fill missing sentiment with neutral values
    sentiment_cols = ['pos', 'neg', 'neu', 'compound', 'sent_count',
                     'pos_surprise', 'neg_surprise', 'compound_surprise']
    
    for col in sentiment_cols:
        if col in merged.columns:
            if col == 'sent_count':
                merged[col] = merged[col].fillna(0)
            elif 'surprise' in col:
                merged[col] = merged[col].fillna(0)
            elif col == 'neu':
                merged[col] = merged[col].fillna(1.0)
            else:
                merged[col] = merged[col].fillna(0.0)
    
    print(f"Merged to {len(merged)} rows")
    return merged


def create_targets(df: pd.DataFrame, 
                  threshold: float = 0.003) -> pd.DataFrame:
    """
    Create prediction targets.
    
    Args:
        df: DataFrame with features
        threshold: Threshold for direction classification (e.g., 0.003 = 0.3%)
    
    Returns:
        DataFrame with targets added
    """
    print("Creating targets...")
    
    targets = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date')
        
        # Next day close (regression target)
        ticker_df['target_close'] = ticker_df['Close'].shift(-1)
        
        # Next day return
        ticker_df['target_return'] = ticker_df['return_1d'].shift(-1)
        
        # Direction (classification target with dead zone)
        ticker_df['target_direction'] = 0  # Neutral
        ticker_df.loc[ticker_df['target_return'] > threshold, 'target_direction'] = 1  # Up
        ticker_df.loc[ticker_df['target_return'] < -threshold, 'target_direction'] = -1  # Down
        
        targets.append(ticker_df)
    
    result = pd.concat(targets, ignore_index=True)
    
    # Remove last row for each ticker (no target available)
    result = result.dropna(subset=['target_close'])
    
    print(f"Created targets for {len(result)} rows")
    return result


def select_features(df: pd.DataFrame, include_sentiment: bool = True) -> Tuple[pd.DataFrame, list]:
    """
    Select final feature set.
    
    Args:
        df: DataFrame with all features
        include_sentiment: Whether to include sentiment features
    
    Returns:
        Tuple of (DataFrame with selected features, list of feature names)
    """
    # Base price features
    price_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'return_1d', 'return_5d',
        'volatility_5d', 'volatility_10d',
        'overnight_return',
        'volume_ratio',
        'price_to_ma5', 'price_to_ma20',
        'rsi',
        'macd', 'macd_signal', 'macd_histogram'
    ]
    
    # Sentiment features
    sentiment_features = [
        'pos', 'neg', 'neu', 'compound', 'sent_count',
        'pos_surprise', 'neg_surprise', 'compound_surprise'
    ]
    
    # Select features
    feature_cols = price_features.copy()
    if include_sentiment:
        # Only add sentiment features that exist
        feature_cols.extend([f for f in sentiment_features if f in df.columns])
    
    # Keep metadata and targets
    meta_cols = ['Date', 'ticker']
    target_cols = ['target_close', 'target_return', 'target_direction']
    
    all_cols = meta_cols + feature_cols + target_cols
    
    # Filter to existing columns
    all_cols = [c for c in all_cols if c in df.columns]
    
    result = df[all_cols].copy()
    
    print(f"Selected {len(feature_cols)} features")
    
    return result, feature_cols


def make_dataset(prices_path: str,
                sentiment_path: Optional[str] = None,
                include_sentiment: bool = True,
                direction_threshold: float = 0.003,
                binary_classification: bool = False,
                train_end: str = '2021-12-31',
                val_end: str = '2023-12-31',
                output_path: str = 'data/processed/dataset.parquet') -> pd.DataFrame:
    """
    Create complete supervised learning dataset.
    
    Args:
        prices_path: Path to price data
        sentiment_path: Path to sentiment data (optional)
        include_sentiment: Whether to include sentiment features
        direction_threshold: Threshold for direction classification
        binary_classification: If True, filter out neutral class (target_direction == 0)
        train_end: End date for training set
        val_end: End date for validation set
        output_path: Output file path
    
    Returns:
        Complete dataset DataFrame
    """
    # Load prices
    prices = load_prices(prices_path)
    
    # Build technical features
    prices = build_technical_features(prices)
    
    # Merge with sentiment if available
    if sentiment_path and Path(sentiment_path).exists() and include_sentiment:
        sentiment = load_sentiment(sentiment_path)
        prices = merge_with_sentiment(prices, sentiment)
    else:
        print("Skipping sentiment features")
        include_sentiment = False
    
    # Create targets
    dataset = create_targets(prices, threshold=direction_threshold)
    
    # Select features
    dataset, feature_cols = select_features(dataset, include_sentiment=include_sentiment)
    
    # Filter out neutral class for binary classification
    if binary_classification:
        print("\nFiltering for binary classification (removing neutral class)...")
        original_len = len(dataset)
        dataset = dataset[dataset['target_direction'] != 0].copy()
        print(f"  Removed {original_len - len(dataset)} neutral samples")
        print(f"  Remaining: {len(dataset)} samples")
        
        # Show class distribution
        print(f"  Class distribution:")
        print(f"    Up (+1):   {(dataset['target_direction'] == 1).sum()}")
        print(f"    Down (-1): {(dataset['target_direction'] == -1).sum()}")
    
    # Remove rows with NaN in features (from rolling windows)
    dataset = dataset.dropna(subset=feature_cols)
    
    # Set Date as index for splitting
    dataset = dataset.set_index('Date')
    
    # Split by date
    train, val, test = split_by_date(dataset, train_end=train_end, val_end=val_end)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train)} rows ({train.index.min()} to {train.index.max()})")
    print(f"  Val:   {len(val)} rows ({val.index.min()} to {val.index.max()})")
    print(f"  Test:  {len(test)} rows ({test.index.min()} to {test.index.max()})")
    
    # Reset index
    dataset = dataset.reset_index()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    
    print(f"\nSaved dataset to {output_path}")
    print(f"Features: {feature_cols}")
    
    # Save feature list
    feature_list_path = output_path.parent / 'feature_list.txt'
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Saved feature list to {feature_list_path}")
    
    return dataset


def main():
    """Command-line interface for making dataset."""
    parser = argparse.ArgumentParser(description='Create supervised learning dataset')
    parser.add_argument('--prices', required=True,
                       help='Path to price data')
    parser.add_argument('--sentiment', default=None,
                       help='Path to sentiment data')
    parser.add_argument('--no_sentiment', action='store_true',
                       help='Exclude sentiment features')
    parser.add_argument('--threshold', type=float, default=0.003,
                       help='Direction classification threshold')
    parser.add_argument('--train_end', default='2021-12-31',
                       help='End date for training set')
    parser.add_argument('--val_end', default='2023-12-31',
                       help='End date for validation set')
    parser.add_argument('--out', default='data/processed/dataset.parquet',
                       help='Output file path')
    
    args = parser.parse_args()
    
    make_dataset(
        prices_path=args.prices,
        sentiment_path=args.sentiment,
        include_sentiment=not args.no_sentiment,
        direction_threshold=args.threshold,
        train_end=args.train_end,
        val_end=args.val_end,
        output_path=args.out
    )


if __name__ == '__main__':
    main()
