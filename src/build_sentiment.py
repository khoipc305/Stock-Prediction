"""
Build sentiment features from news/Reddit data using VADER and optional FinBERT.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# VADER sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional: FinBERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

from .utils import align_to_market_day


class SentimentAnalyzer:
    """Sentiment analyzer supporting VADER and FinBERT."""
    
    def __init__(self, method: str = 'vader'):
        """
        Initialize sentiment analyzer.
        
        Args:
            method: 'vader' or 'finbert'
        """
        self.method = method
        
        if method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        elif method == 'finbert':
            if not FINBERT_AVAILABLE:
                raise ImportError("transformers not installed. Install with: pip install transformers")
            
            print("Loading FinBERT model...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.analyzer = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'vader' or 'finbert'")
    
    def score_text(self, text: str) -> dict:
        """
        Score sentiment of a single text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment scores
        """
        if pd.isna(text) or not text.strip():
            return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
        
        if self.method == 'vader':
            scores = self.analyzer.polarity_scores(text)
            return scores
        
        elif self.method == 'finbert':
            # FinBERT returns label and score
            try:
                result = self.analyzer(text[:512])[0]  # Limit to 512 tokens
                label = result['label'].lower()
                score = result['score']
                
                # Convert to VADER-like format
                if label == 'positive':
                    return {'pos': score, 'neg': 0.0, 'neu': 1-score, 'compound': score}
                elif label == 'negative':
                    return {'pos': 0.0, 'neg': score, 'neu': 1-score, 'compound': -score}
                else:  # neutral
                    return {'pos': 0.0, 'neg': 0.0, 'neu': score, 'compound': 0.0}
            except Exception as e:
                print(f"Error scoring text: {e}")
                return {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}


def clean_text(text: str) -> str:
    """
    Clean text data.
    
    Args:
        text: Input text
    
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    import re
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def load_kaggle_data(csv_path: str) -> pd.DataFrame:
    """
    Load Kaggle news/Reddit CSV.
    
    Supports multiple formats:
    1. Standard: timestamp, ticker, text, weight (optional)
    2. FinViz style: ticker, date, time, headline
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"Loading Kaggle data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check for FinViz-style format (ticker, date, time, headline)
    if 'headline' in df.columns and 'date' in df.columns and 'time' in df.columns:
        print("Detected FinViz-style format (ticker, date, time, headline)")
        
        # Combine date and time into timestamp
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
        
        # Rename headline to text
        df['text'] = df['headline']
        
        # Add weight if not present
        if 'weight' not in df.columns:
            df['weight'] = 1.0
        
        # Keep only needed columns
        df = df[['timestamp', 'ticker', 'text', 'weight']].copy()
    
    # Check for standard format
    elif 'text' in df.columns or 'headline' in df.columns:
        # Use 'headline' as 'text' if text doesn't exist
        if 'text' not in df.columns and 'headline' in df.columns:
            df['text'] = df['headline']
        
        # Ensure required columns exist
        required_cols = ['ticker', 'text']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Handle timestamp column
        if 'timestamp' not in df.columns:
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                raise ValueError("Missing timestamp or date column")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Add weight if not present
        if 'weight' not in df.columns:
            df['weight'] = 1.0
    else:
        raise ValueError("Unrecognized CSV format. Expected columns: timestamp/date, ticker, text/headline")
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Tickers: {df['ticker'].unique()[:10]}...")  # Show first 10 tickers
    
    return df


def load_reddit_data(csv_path: str) -> pd.DataFrame:
    """
    Load Reddit dump CSV.
    
    Expected columns: created_utc, subreddit, title, selftext, score
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with cleaned data
    """
    print(f"Loading Reddit data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Combine title and selftext
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    df['text'] = df['text'].apply(clean_text)
    
    # Convert timestamp
    if 'created_utc' in df.columns:
        df['timestamp'] = pd.to_datetime(df['created_utc'], unit='s')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("No timestamp column found")
    
    # Use score as weight
    if 'score' in df.columns:
        df['weight'] = df['score'].clip(lower=1)
    else:
        df['weight'] = 1.0
    
    # Extract ticker from text or use a default
    # This is simplified - you may need more sophisticated ticker extraction
    if 'ticker' not in df.columns:
        df['ticker'] = 'UNKNOWN'
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    print(f"Loaded {len(df)} records")
    return df


def score_sentiment(df: pd.DataFrame, method: str = 'vader', batch_size: int = 100) -> pd.DataFrame:
    """
    Score sentiment for all texts.
    
    Args:
        df: DataFrame with 'text' column
        method: 'vader' or 'finbert'
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with sentiment scores added
    """
    print(f"Scoring sentiment using {method.upper()}...")
    
    analyzer = SentimentAnalyzer(method=method)
    
    scores = []
    for text in tqdm(df['text'], desc="Scoring"):
        score = analyzer.score_text(text)
        scores.append(score)
    
    # Add scores to dataframe
    df['pos'] = [s['pos'] for s in scores]
    df['neg'] = [s['neg'] for s in scores]
    df['neu'] = [s['neu'] for s in scores]
    df['compound'] = [s['compound'] for s in scores]
    
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aggregate sentiment to daily per-ticker level.
    
    Args:
        df: DataFrame with sentiment scores and market_day
        tickers: Optional list of tickers to filter
    
    Returns:
        Daily aggregated sentiment DataFrame
    """
    print("Aggregating sentiment to daily level...")
    
    # Align to market day
    df['market_day'] = df['timestamp'].apply(align_to_market_day)
    
    # Filter tickers if specified
    if tickers:
        df = df[df['ticker'].isin(tickers)]
    
    # Weighted aggregation
    agg_dict = {
        'pos': lambda x: np.average(x, weights=df.loc[x.index, 'weight']),
        'neg': lambda x: np.average(x, weights=df.loc[x.index, 'weight']),
        'neu': lambda x: np.average(x, weights=df.loc[x.index, 'weight']),
        'compound': lambda x: np.average(x, weights=df.loc[x.index, 'weight']),
        'text': 'count'  # Count of texts
    }
    
    daily = df.groupby(['ticker', 'market_day']).agg(agg_dict).reset_index()
    daily.rename(columns={'text': 'sent_count'}, inplace=True)
    
    # Calculate surprise features (deviation from 20-day average)
    for col in ['pos', 'neg', 'compound']:
        daily[f'{col}_ma20'] = daily.groupby('ticker')[col].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        daily[f'{col}_surprise'] = daily[col] - daily[f'{col}_ma20']
    
    print(f"Aggregated to {len(daily)} daily records")
    return daily


def build_sentiment(kaggle_csv: Optional[str] = None,
                   reddit_csv: Optional[str] = None,
                   method: str = 'vader',
                   tickers: Optional[List[str]] = None,
                   output_path: str = 'data/interim/sentiment_daily.parquet') -> pd.DataFrame:
    """
    Build sentiment features from multiple sources.
    
    Args:
        kaggle_csv: Path to Kaggle CSV
        reddit_csv: Path to Reddit CSV
        method: Sentiment method ('vader' or 'finbert')
        tickers: List of tickers to filter
        output_path: Output file path
    
    Returns:
        Daily sentiment DataFrame
    """
    all_data = []
    
    # Load Kaggle data
    if kaggle_csv and Path(kaggle_csv).exists():
        kaggle_df = load_kaggle_data(kaggle_csv)
        kaggle_df = score_sentiment(kaggle_df, method=method)
        all_data.append(kaggle_df)
    
    # Load Reddit data
    if reddit_csv and Path(reddit_csv).exists():
        reddit_df = load_reddit_data(reddit_csv)
        reddit_df = score_sentiment(reddit_df, method=method)
        all_data.append(reddit_df)
    
    if not all_data:
        raise ValueError("No data sources provided or found")
    
    # Combine all sources
    combined = pd.concat(all_data, ignore_index=True)
    
    # Aggregate to daily
    daily_sentiment = aggregate_daily_sentiment(combined, tickers=tickers)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily_sentiment.to_parquet(output_path, index=False)
    
    print(f"\nSaved daily sentiment to {output_path}")
    print(f"Date range: {daily_sentiment['market_day'].min()} to {daily_sentiment['market_day'].max()}")
    
    return daily_sentiment


def main():
    """Command-line interface for building sentiment."""
    parser = argparse.ArgumentParser(description='Build sentiment features from news/Reddit data')
    parser.add_argument('--kaggle_csv', default=None,
                       help='Path to Kaggle news CSV')
    parser.add_argument('--reddit_csv', default=None,
                       help='Path to Reddit dump CSV')
    parser.add_argument('--method', default='vader', choices=['vader', 'finbert'],
                       help='Sentiment analysis method')
    parser.add_argument('--tickers', nargs='*', default=None,
                       help='Filter to specific tickers')
    parser.add_argument('--out', default='data/interim/sentiment_daily.parquet',
                       help='Output file path')
    
    args = parser.parse_args()
    
    build_sentiment(
        kaggle_csv=args.kaggle_csv,
        reddit_csv=args.reddit_csv,
        method=args.method,
        tickers=args.tickers,
        output_path=args.out
    )


if __name__ == '__main__':
    main()
