# Stock Price Prediction Deployment

A web-based application for predicting stock prices using a trained LSTM model.

## 🚀 Features

- **Real-time Stock Data**: Fetches live stock data from Yahoo Finance
- **LSTM Predictions**: Uses trained deep learning model for price forecasting
- **Interactive Charts**: Visualize historical data and predictions
- **Multiple Stocks**: Support for any stock ticker available on Yahoo Finance
- **Confidence Intervals**: Shows prediction uncertainty
- **Technical Indicators**: Includes RSI, MACD, moving averages, and more

## 📋 Prerequisites

Before running the application, ensure you have:

1. Python 3.8 or higher
2. Trained LSTM model file (`lstm_early_fusion.pt`)
3. Required Python packages (see requirements.txt)

## 🛠️ Installation

1. **Navigate to deployment directory**:
   ```bash
   cd deployment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy trained model**:
   ```bash
   # Copy your trained model to the models directory
   copy ..\models\lstm_early_fusion.pt models\
   ```

## 🎯 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Application

1. **Enter Stock Ticker**: Type any valid stock symbol (e.g., AAPL, GOOGL, MSFT)
2. **Select Date Range**: Choose the historical data period
3. **Set Forecast Days**: Select how many days to predict (1-30)
4. **Generate Prediction**: Click the button to see results

### Example Stocks to Try

- **AAPL** - Apple Inc.
- **GOOGL** - Alphabet Inc.
- **MSFT** - Microsoft Corporation
- **TSLA** - Tesla Inc.
- **AMZN** - Amazon.com Inc.
- **NVDA** - NVIDIA Corporation

## 📊 Model Information

- **Architecture**: LSTM (Long Short-Term Memory)
- **Input Features**: 25 features including:
  - Price data (OHLCV)
  - Technical indicators (RSI, MACD, Moving Averages)
  - Sentiment features (from news analysis)
- **Training Period**: 2015-2021
- **Lookback Window**: 30 days
- **Output**: Next-day price prediction

## 🔧 Configuration

Edit `config.py` to customize:

- Model parameters
- Feature list
- Default settings
- API keys (for news sentiment)

## 📁 Project Structure

```
deployment/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── models/               # Trained model files
│   └── lstm_early_fusion.pt
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── data_fetcher.py   # Stock data fetching
│   ├── preprocessor.py   # Feature engineering
│   └── predictor.py      # Prediction logic
├── static/               # Static assets (CSS, images)
└── templates/            # HTML templates (if using Flask)
```

## ⚠️ Disclaimer

**This application is for educational purposes only.**

- Predictions are based on historical data and machine learning models
- Past performance does not guarantee future results
- Do NOT use these predictions as the sole basis for investment decisions
- Always consult with a qualified financial advisor before making investment decisions
- The creators are not responsible for any financial losses

## 🐛 Troubleshooting

### Model Not Found Error
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/lstm_early_fusion.pt'
```
**Solution**: Copy your trained model to the `deployment/models/` directory

### Data Fetching Error
```
Could not fetch data for TICKER
```
**Solution**: 
- Check if the ticker symbol is valid
- Ensure you have internet connection
- Try a different date range

### Import Error
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

## 🔮 Future Enhancements

- [ ] Add more technical indicators
- [ ] Integrate real-time news sentiment
- [ ] Support for multiple stocks comparison
- [ ] Portfolio optimization features
- [ ] Export predictions to CSV
- [ ] Email alerts for price targets
- [ ] Mobile-responsive design
- [ ] REST API endpoint

## 📝 License

This project is part of the CS4200 Stock Prediction coursework.

## 👥 Contributors

- Your Name
- CS4200 - Cal Poly Pomona

## 📧 Contact

For questions or issues, please contact [your-email@example.com]

---

**Happy Predicting! 📈**
