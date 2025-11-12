# 🚀 Quick Start Guide

Get your stock prediction app running in 5 minutes!

## Step 1: Install Dependencies

```bash
cd deployment
pip install -r requirements.txt
```

## Step 2: Setup Environment

```bash
python setup.py
```

This will:
- Create necessary directories
- Copy your trained model
- Check if all packages are installed

## Step 3: Run the Application

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

## Step 4: Make Your First Prediction

1. Enter a stock ticker (e.g., **AAPL**)
2. Click **"Generate Prediction"**
3. View the results!

## 🎯 What You Can Do

### Predict Any Stock
- Try: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA
- Works with any ticker on Yahoo Finance

### Adjust Forecast Period
- Slide to predict 1-30 days ahead
- See confidence intervals

### View Technical Analysis
- RSI, MACD, Moving Averages
- Volume analysis
- Price trends

## 📊 Understanding the Results

### Prediction Card
- **Predicted Price**: Model's forecast
- **Change**: Difference from current price
- **Direction**: UP 📈 or DOWN 📉

### Chart
- **Blue Line**: Historical prices
- **Orange Dashed**: Predictions
- **Shaded Area**: Confidence interval

### Metrics
- **Current Price**: Latest closing price
- **Volume**: Trading volume
- **Data Points**: Days of historical data used

## ⚡ Pro Tips

1. **More Data = Better Predictions**
   - Use at least 1 year of historical data
   - Model needs 30 days minimum

2. **Check Multiple Timeframes**
   - Compare 1-day, 5-day, and 30-day forecasts
   - Look for consistent trends

3. **Consider Confidence Intervals**
   - Wider intervals = more uncertainty
   - Narrow intervals = more confident prediction

4. **Use Technical Indicators**
   - RSI > 70: Overbought
   - RSI < 30: Oversold
   - MACD crossovers signal trend changes

## 🔧 Troubleshooting

### App Won't Start
```bash
# Make sure you're in the deployment directory
cd deployment

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Try running again
streamlit run app.py
```

### Model Not Found
```bash
# Run setup script
python setup.py

# Or manually copy model
copy ..\models\lstm_early_fusion.pt models\
```

### Can't Fetch Stock Data
- Check your internet connection
- Verify ticker symbol is correct
- Try a different stock

## 📱 Accessing from Other Devices

### On Your Network
1. Find your IP address:
   ```bash
   ipconfig  # Windows
   ifconfig  # Mac/Linux
   ```

2. Share this URL with others on your network:
   ```
   http://YOUR_IP_ADDRESS:8501
   ```

### Deploy to Cloud (Optional)
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Deploy as web app
- **AWS/Azure**: For production use

## 🎓 Next Steps

1. **Customize the App**
   - Edit `app.py` to add features
   - Modify `config.py` for settings

2. **Improve the Model**
   - Retrain with more data
   - Try different architectures
   - Add more features

3. **Add More Features**
   - Portfolio tracking
   - Email alerts
   - Multiple stock comparison

## ⚠️ Important Reminders

- This is for **educational purposes only**
- Do NOT use for actual trading decisions
- Always consult financial advisors
- Past performance ≠ future results

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Review error messages carefully
- Ensure all dependencies are installed
- Verify model file exists in `models/` directory

---

**Ready to predict? Let's go! 🚀**

```bash
streamlit run app.py
```
