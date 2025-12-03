"""
Configuration settings for stock prediction deployment
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# Model settings
DEFAULT_MODEL = 'lstm_early_fusion.pt'
LOOKBACK_PERIOD = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# Data settings
DEFAULT_TICKER = 'AAPL'
DEFAULT_HISTORY_DAYS = 365
MAX_FORECAST_DAYS = 30

# Feature list (must match training)
FEATURE_LIST = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'return_1d', 'return_5d', 'volatility_5d', 'volatility_10d',
    'overnight_return', 'volume_ratio', 'price_to_ma5', 'price_to_ma20',
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'pos', 'neg', 'neu', 'compound', 'sent_count',
    'pos_surprise', 'neg_surprise', 'compound_surprise'
]

# API settings (for future use)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')

# UI settings
APP_TITLE = "Stock Price Predictor"
APP_ICON = "📈"
THEME = "light"
