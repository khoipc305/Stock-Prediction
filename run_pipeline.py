"""
Complete pipeline script to run the entire stock prediction workflow.

This script:
1. Downloads price data
2. Builds the dataset (price-only, no sentiment)
3. Trains the model
4. Evaluates and backtests

For sentiment data, use your FinViz scraper and process_finviz_data.py separately.

Usage:
    python run_pipeline.py --tickers MSFT AAPL NVDA --quick
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.download_prices import download_prices
from src.make_dataset import make_dataset
from src.train_lstm import train_model
from src.evaluate import evaluate_model, simple_backtest


# Sample sentiment generation removed - use real FinViz data instead
# See process_finviz_data.py for processing real FinViz headlines


def run_pipeline(tickers, start_date='2015-01-01', quick=False):
    """
    Run the complete pipeline.
    
    Args:
        tickers: List of stock tickers
        start_date: Start date for data
        quick: If True, use reduced epochs for faster training
    """
    print("\n" + "="*60)
    print("STOCK PREDICTION PIPELINE")
    print("="*60)
    print(f"Tickers: {tickers}")
    print(f"Start Date: {start_date}")
    print(f"Quick Mode: {quick}")
    
    # Step 1: Download prices
    print("\n" + "="*60)
    print("STEP 1: Downloading Price Data")
    print("="*60)
    
    prices_path = 'data/raw/prices.parquet'
    if not Path(prices_path).exists():
        download_prices(
            tickers=tickers,
            start_date=start_date,
            output_path=prices_path
        )
    else:
        print(f"Price data already exists at {prices_path}")
    
    # Step 2: Check for sentiment data (optional)
    print("\n" + "="*60)
    print("STEP 2: Checking for Sentiment Data")
    print("="*60)
    
    sentiment_path = 'data/interim/sentiment_daily.parquet'
    include_sentiment = Path(sentiment_path).exists()
    
    if include_sentiment:
        print(f"✓ Found sentiment data at {sentiment_path}")
        print("  Will train model with sentiment features")
    else:
        print(f"⚠ No sentiment data found at {sentiment_path}")
        print("  Will train price-only model")
        print("\nTo add sentiment:")
        print("  1. Export FinViz data from your sample notebook")
        print("  2. Run: python process_finviz_data.py --input data/raw/finviz_news.csv")
        sentiment_path = None
    
    # Step 3: Create dataset
    print("\n" + "="*60)
    print("STEP 3: Creating Supervised Learning Dataset")
    print("="*60)
    
    dataset_path = 'data/processed/dataset.parquet'
    make_dataset(
        prices_path=prices_path,
        sentiment_path=sentiment_path,
        include_sentiment=include_sentiment,
        output_path=dataset_path
    )
    
    # Step 4: Train model
    print("\n" + "="*60)
    print("STEP 4: Training LSTM Model")
    print("="*60)
    
    epochs = 20 if quick else 60
    model_path = 'models/best_lstm.pt'
    
    train_model(
        dataset_path=dataset_path,
        lookback=30,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        learning_rate=0.001,
        epochs=epochs,
        patience=8,
        model_type='early_fusion',
        save_path=model_path,
        seed=42
    )
    
    # Step 5: Evaluate
    print("\n" + "="*60)
    print("STEP 5: Evaluating Model")
    print("="*60)
    
    evaluate_model(
        dataset_path=dataset_path,
        model_path=model_path,
        output_dir='reports/figures'
    )
    
    # Step 6: Backtest
    print("\n" + "="*60)
    print("STEP 6: Running Backtest")
    print("="*60)
    
    simple_backtest(
        dataset_path=dataset_path,
        model_path=model_path,
        strategy='long',
        transaction_cost=0.001,
        output_dir='reports/figures'
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nResults saved to:")
    print("  - Models: models/")
    print("  - Figures: reports/figures/")
    print("  - Data: data/processed/")
    print("\nCheck the notebooks for detailed analysis.")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Run the complete stock prediction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default tickers (quick mode)
  python run_pipeline.py --quick
  
  # Run with custom tickers
  python run_pipeline.py --tickers MSFT AAPL GOOGL AMZN
  
  # Full training (takes longer)
  python run_pipeline.py --tickers MSFT AAPL NVDA
        """
    )
    
    parser.add_argument(
        '--tickers',
        nargs='+',
        default=['MSFT', 'AAPL', 'NVDA'],
        help='Stock ticker symbols (default: MSFT AAPL NVDA)'
    )
    
    parser.add_argument(
        '--start',
        default='2015-01-01',
        help='Start date for data (default: 2015-01-01)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode with reduced epochs for testing'
    )
    
    args = parser.parse_args()
    
    # Download VADER lexicon if needed
    try:
        import nltk
        nltk.data.find('vader_lexicon')
    except LookupError:
        print("Downloading VADER lexicon...")
        import nltk
        nltk.download('vader_lexicon')
    
    # Run pipeline
    run_pipeline(
        tickers=args.tickers,
        start_date=args.start,
        quick=args.quick
    )


if __name__ == '__main__':
    main()
