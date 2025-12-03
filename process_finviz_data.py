"""
Process FinViz-style sentiment data (like from the sample notebook).

This script is specifically designed to work with CSV files that have the format:
ticker, date, time, headline

Usage:
    python process_finviz_data.py --input "codes sample/finviz_data.csv" --output data/interim/sentiment_daily.parquet
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.build_sentiment import build_sentiment


def process_finviz_csv(input_csv: str, output_path: str = 'data/interim/sentiment_daily.parquet'):
    """
    Process a FinViz-style CSV file.
    
    Expected format: ticker, date, time, headline
    
    Args:
        input_csv: Path to input CSV file
        output_path: Path to save processed sentiment data
    """
    print("="*60)
    print("PROCESSING FINVIZ-STYLE DATA")
    print("="*60)
    
    # Check if file exists
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"Error: File not found: {input_csv}")
        print("\nExpected format:")
        print("  ticker, date, time, headline")
        print("\nExample:")
        print("  MCD,2021-07-28,09:30AM,McDonald's Reports Strong Earnings")
        print("  MCD,2021-07-28,10:15AM,Fast Food Giant Beats Expectations")
        return
    
    # Load the CSV
    print(f"\nLoading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Display info
    print(f"\nColumns found: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Check format
    required_cols = ['ticker', 'date', 'time', 'headline']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nError: Missing required columns: {missing_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        print("\nExpected format: ticker, date, time, headline")
        return
    
    # Process with build_sentiment
    print("\n" + "="*60)
    print("BUILDING SENTIMENT FEATURES")
    print("="*60)
    
    daily_sentiment = build_sentiment(
        kaggle_csv=input_csv,
        method='vader',
        output_path=output_path
    )
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"\nOutput saved to: {output_path}")
    print(f"Records: {len(daily_sentiment)}")
    print(f"Tickers: {daily_sentiment['ticker'].unique().tolist()}")
    print(f"Date range: {daily_sentiment['market_day'].min()} to {daily_sentiment['market_day'].max()}")
    
    return daily_sentiment


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Process FinViz-style sentiment data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process FinViz CSV
  python process_finviz_data.py --input "codes sample/mcd_news.csv"
  
  # Specify output location
  python process_finviz_data.py --input data/raw/finviz_news.csv --output data/interim/sentiment_daily.parquet
  
Expected CSV format:
  ticker,date,time,headline
  MCD,2021-07-28,09:30AM,McDonald's Reports Strong Earnings
  MCD,2021-07-28,10:15AM,Fast Food Giant Beats Expectations
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Path to input CSV file (FinViz format: ticker, date, time, headline)'
    )
    
    parser.add_argument(
        '--output',
        default='data/interim/sentiment_daily.parquet',
        help='Path to save processed sentiment data (default: data/interim/sentiment_daily.parquet)'
    )
    
    args = parser.parse_args()
    
    # Process the data
    process_finviz_csv(args.input, args.output)


if __name__ == '__main__':
    main()
