"""
Download historical stock prices from Yahoo Finance.
"""

import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import List
from tqdm import tqdm
from .safe_file_utils import safe_to_parquet


def download_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download price data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"Warning: No data found for {ticker}")
            return pd.DataFrame()
        
        # Flatten multi-index columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        return data
    
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()


def download_prices(tickers: List[str], 
                   start_date: str = '2015-01-01',
                   end_date: str = None,
                   output_path: str = 'data/raw/prices.parquet') -> pd.DataFrame:
    """
    Download price data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        output_path: Path to save the data
    
    Returns:
        Combined DataFrame with all tickers
    """
    if end_date is None:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    print(f"Downloading price data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    all_data = []
    
    for ticker in tqdm(tickers, desc="Downloading"):
        df = download_ticker(ticker, start_date, end_date)
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No data downloaded for any ticker")
    
    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Ensure Date column is datetime
    combined['Date'] = pd.to_datetime(combined['Date'])
    
    # Sort by ticker and date
    combined = combined.sort_values(['ticker', 'Date']).reset_index(drop=True)
    
    # Save to file (use safe_to_parquet to avoid OneDrive sync issues)
    safe_to_parquet(combined, output_path, index=False)
    
    print(f"\nSaved {len(combined)} rows to {output_path}")
    print(f"Tickers: {combined['ticker'].unique().tolist()}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    
    return combined


def main():
    """Command-line interface for downloading prices."""
    parser = argparse.ArgumentParser(description='Download stock prices from Yahoo Finance')
    parser.add_argument('--tickers', nargs='+', required=True, 
                       help='Stock ticker symbols (e.g., MSFT AAPL NVDA)')
    parser.add_argument('--start', default='2015-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=None,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--out', default='data/raw/prices.parquet',
                       help='Output file path')
    
    args = parser.parse_args()
    
    download_prices(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        output_path=args.out
    )


if __name__ == '__main__':
    main()
