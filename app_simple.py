"""
Stock Prediction Web Application
Streamlit-based interface for LSTM stock price prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetcher import fetch_stock_data
from utils.preprocessor import prepare_features
from utils.predictor import StockPredictor

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize predictor
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        return StockPredictor(model_path='models/lstm_early_fusion.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.title("Stock Price Predictor")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Stock ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_range = st.date_input(
            "Historical Data Range",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        # Prediction horizon
        forecast_days = st.slider(
            "Forecast Days",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of days to predict into the future"
        )
        
        # Predict button
        predict_button = st.button("Generate Prediction", type="primary", use_container_width=True)
    
    # Main content
    if predict_button:
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                # Fetch data
                data = fetch_stock_data(ticker, date_range[0], date_range[1])
                
                if data is None or len(data) == 0:
                    st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
                    return
                
                st.success(f"Successfully fetched {len(data)} days of data")
                
                # Display current price
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2]
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
                    )
                with col2:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                with col3:
                    st.metric("Data Points", len(data))
                
                # Load predictor and make predictions
                with st.spinner("Generating predictions..."):
                    predictor = load_predictor()
                    if predictor is None:
                        st.error("Failed to load prediction model. Please check the logs.")
                        return
                    predictions = predictor.predict(data, forecast_days)
                
                # Display predictions
                st.subheader("Predictions")
                
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    st.markdown("### Next Day Prediction")
                    next_day_price = predictions['prices'][0]
                    next_day_change = next_day_price - current_price
                    next_day_change_pct = (next_day_change / current_price) * 100
                    
                    st.metric(
                        "Predicted Price",
                        f"${next_day_price:.2f}",
                        f"{next_day_change:+.2f} ({next_day_change_pct:+.2f}%)"
                    )
                    
                    direction = "UP" if next_day_change > 0 else "DOWN"
                    st.markdown(f"**Direction:** {direction}")
                
                with pred_col2:
                    st.markdown(f"### {forecast_days}-Day Forecast")
                    final_price = predictions['prices'][-1]
                    total_change = final_price - current_price
                    total_change_pct = (total_change / current_price) * 100
                    
                    st.metric(
                        f"Price in {forecast_days} days",
                        f"${final_price:.2f}",
                        f"{total_change:+.2f} ({total_change_pct:+.2f}%)"
                    )
                
                # Plot predictions
                st.subheader("Price Chart with Predictions")
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=data.index[-60:],
                    y=data['Close'].iloc[-60:],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Predictions
                pred_dates = pd.date_range(
                    start=data.index[-1] + timedelta(days=1),
                    periods=forecast_days,
                    freq='D'
                )
                
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions['prices'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                
                # Confidence intervals (if available)
                if 'confidence_lower' in predictions and 'confidence_upper' in predictions:
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=predictions['confidence_upper'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=predictions['confidence_lower'],
                        mode='lines',
                        name='Confidence Interval',
                        fill='tonexty',
                        fillcolor='rgba(255, 127, 14, 0.2)',
                        line=dict(width=0)
                    ))
                
                fig.update_layout(
                    title=f"{ticker} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode='x unified',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction table
                st.subheader("Detailed Predictions")
                pred_df = pd.DataFrame({
                    'Date': pred_dates,
                    'Predicted Price': [f"${p:.2f}" for p in predictions['prices']],
                    'Change from Current': [f"{(p - current_price):+.2f}" for p in predictions['prices']],
                    'Change %': [f"{((p - current_price) / current_price * 100):+.2f}%" for p in predictions['prices']]
                })
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
                
                # Model info
                with st.expander("Model Information"):
                    st.markdown("""
                    **Model Type:** LSTM (Long Short-Term Memory)
                    
                    **Features Used:**
                    - Price data (Open, High, Low, Close, Volume)
                    - Technical indicators (RSI, MACD, Moving Averages)
                    - Sentiment analysis from news headlines
                    
                    **Training Period:** 2015-2021
                    
                    **Validation Accuracy:** ~50% direction accuracy
                    
                    **Disclaimer:** This is a machine learning model trained on historical data. 
                    Predictions should not be used as the sole basis for investment decisions.
                    """)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    else:
        # Welcome message
        st.info("""
        **Get Started:**
        1. Enter a stock ticker symbol in the sidebar
        2. Select the date range for historical data
        3. Choose how many days to forecast
        4. Click "Generate Prediction" to see results
        
        **Supported Stocks:** Any ticker available on Yahoo Finance (e.g., AAPL, GOOGL, MSFT, TSLA, etc.)
        """)

if __name__ == "__main__":
    main()
