"""
Data fetching utilities for real-time stock data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date=None, end_date=None):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    start_date : datetime or str
        Start date for historical data
    end_date : datetime or str
        End date for historical data
    
    Returns:
    --------
    pd.DataFrame
        Stock price data with OHLCV columns
    """
    try:
        # Default to last year if no dates provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        # Clean data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = data.dropna()
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def fetch_latest_price(ticker):
    """
    Fetch the latest price for a stock
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    
    Returns:
    --------
    float
        Latest closing price
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d')
        return data['Close'].iloc[-1]
    except:
        return None

def fetch_news_sentiment(ticker, days=7):
    """
    Fetch recent news and sentiment for a stock
    (Placeholder - would need API key for real implementation)
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days : int
        Number of days of news to fetch
    
    Returns:
    --------
    pd.DataFrame
        News headlines with sentiment scores
    """
    # This is a placeholder
    # In production, you would use:
    # - NewsAPI
    # - Alpha Vantage
    # - Finnhub
    # - Twitter API
    
    return pd.DataFrame({
        'date': [],
        'headline': [],
        'sentiment': []
    })
