# 🧪 Testing Guide - Stock Prediction App

## 🚀 Quick Start

### 1. **Access the App**
Open your browser and go to: **http://localhost:8501**

Or click the browser preview button in your IDE!

---

## 📋 **Step-by-Step Testing**

### **Test 1: Basic Prediction (Apple Stock)**

1. **Open the app** in your browser
2. In the **sidebar**, you'll see:
   - Stock Ticker input box
   - Date range selector
   - Forecast days slider
   - "Generate Prediction" button

3. **Enter ticker:** Type `AAPL` (or leave default)
4. **Click:** "🔮 Generate Prediction" button
5. **Wait:** 3-5 seconds for data fetching and prediction

#### ✅ **Expected Results:**
- Current price displayed (e.g., $180-190)
- Next-day prediction with change %
- 5-day forecast chart
- Interactive Plotly chart with:
  - Blue line (historical prices)
  - Orange dashed line (predictions)
  - Shaded area (confidence interval)
- Detailed prediction table

---

### **Test 2: Different Stock (Google)**

1. **Clear the ticker box**
2. **Type:** `GOOGL`
3. **Click:** "Generate Prediction"

#### ✅ **Expected Results:**
- Different price range (~$140-150)
- New predictions
- Chart updates with GOOGL data

---

### **Test 3: Longer Forecast**

1. **Use ticker:** `MSFT`
2. **Adjust slider:** Set forecast to **10 days**
3. **Click:** "Generate Prediction"

#### ✅ **Expected Results:**
- 10-day forecast displayed
- Longer orange prediction line
- Table shows 10 rows of predictions

---

### **Test 4: Different Date Range**

1. **Use ticker:** `TSLA`
2. **Click date range selector**
3. **Select:** Last 6 months (instead of 1 year)
4. **Set forecast:** 7 days
5. **Click:** "Generate Prediction"

#### ✅ **Expected Results:**
- Predictions based on 6 months of data
- Chart shows shorter historical period

---

### **Test 5: Multiple Stocks Comparison**

Try these different stocks to see variety:

| Ticker | Company | Expected Price Range |
|--------|---------|---------------------|
| AAPL | Apple | $170-195 |
| GOOGL | Google | $135-155 |
| MSFT | Microsoft | $370-420 |
| TSLA | Tesla | $200-280 |
| AMZN | Amazon | $170-190 |
| NVDA | NVIDIA | $130-150 |
| META | Meta | $480-560 |

For each:
1. Enter ticker
2. Click "Generate Prediction"
3. Observe different predictions

---

## 🎯 **What to Look For**

### **✅ Good Signs:**
- App loads without errors
- Data fetches successfully
- Predictions generate in 3-5 seconds
- Charts are interactive (hover, zoom, pan)
- Predictions seem reasonable (not crazy numbers)
- Direction signal shows (📈 UP or 📉 DOWN)
- Confidence intervals displayed

### **⚠️ Potential Issues:**
- "Could not fetch data" → Check ticker spelling
- Very long wait → Check internet connection
- Predictions seem off → Normal (model is ~50% accurate)
- Chart doesn't load → Refresh page

---

## 🔍 **Detailed Feature Testing**

### **Sidebar Controls:**

#### 1. **Stock Ticker Input**
- ✅ Test: Enter valid ticker (AAPL)
- ✅ Test: Enter invalid ticker (XYZ123)
- ✅ Test: Lowercase ticker (aapl → converts to AAPL)

#### 2. **Date Range Selector**
- ✅ Test: Default (last year)
- ✅ Test: Last 6 months
- ✅ Test: Last 2 years
- ⚠️ Note: Need minimum 30 days for model

#### 3. **Forecast Days Slider**
- ✅ Test: 1 day (minimum)
- ✅ Test: 5 days (default)
- ✅ Test: 30 days (maximum)

---

### **Main Panel Features:**

#### 1. **Metrics Cards**
Check for:
- ✅ Current Price (with $ symbol)
- ✅ Price change (+ or - with %)
- ✅ Volume (formatted with commas)
- ✅ Data points count

#### 2. **Prediction Cards**

**Next Day Prediction:**
- ✅ Predicted price
- ✅ Change from current
- ✅ Change percentage
- ✅ Direction indicator (📈 or 📉)

**Multi-Day Forecast:**
- ✅ Final predicted price
- ✅ Total change
- ✅ Total change percentage

#### 3. **Interactive Chart**
Test these features:
- ✅ **Hover:** Shows exact values
- ✅ **Zoom:** Click and drag to zoom
- ✅ **Pan:** Shift + drag to pan
- ✅ **Reset:** Double-click to reset view
- ✅ **Legend:** Click to hide/show lines
- ✅ **Download:** Camera icon to save image

#### 4. **Prediction Table**
Check:
- ✅ Date column
- ✅ Predicted Price (formatted)
- ✅ Change from Current
- ✅ Change % column

#### 5. **Model Information**
Expand "ℹ️ Model Information":
- ✅ Model type (LSTM)
- ✅ Features used
- ✅ Training period
- ✅ Disclaimer

---

## 🧪 **Advanced Testing**

### **Test 6: Edge Cases**

#### Invalid Ticker:
```
Ticker: INVALID123
Expected: Error message "Could not fetch data"
```

#### Very Short History:
```
Ticker: AAPL
Date Range: Last 7 days
Expected: Error (need 30+ days)
```

#### Maximum Forecast:
```
Ticker: AAPL
Forecast: 30 days
Expected: Long prediction line, wider confidence
```

---

### **Test 7: Performance**

Time these operations:

| Action | Expected Time |
|--------|--------------|
| App startup | 5-10 seconds |
| First prediction | 5-8 seconds |
| Subsequent predictions | 3-5 seconds |
| Chart interaction | Instant |

---

### **Test 8: UI Responsiveness**

1. **Resize browser window**
   - ✅ Layout adjusts
   - ✅ Chart resizes
   - ✅ Sidebar remains accessible

2. **Mobile view** (narrow window)
   - ✅ Sidebar collapses
   - ✅ Content stacks vertically

---

## 📊 **Sample Test Results**

### **Example: AAPL Test**

```
Input:
- Ticker: AAPL
- Date Range: Last 365 days
- Forecast: 5 days

Output:
- Current Price: $189.50
- Next Day Prediction: $190.25 (+$0.75, +0.40%)
- Direction: 📈 UP
- 5-Day Forecast: $191.80 (+$2.30, +1.21%)

Chart:
- Historical: 365 days of data
- Predictions: 5 orange points
- Confidence: Shaded area around predictions

Status: ✅ PASS
```

---

## 🐛 **Troubleshooting**

### **Problem: App won't start**
```bash
# Solution 1: Check if already running
# Look for "You can now view your Streamlit app in your browser"

# Solution 2: Kill existing process and restart
Ctrl+C (in terminal)
py -m streamlit run app.py
```

### **Problem: Can't fetch data**
```
Possible causes:
1. Invalid ticker symbol
2. No internet connection
3. Yahoo Finance API down
4. Ticker delisted/suspended

Solution: Try a different ticker (AAPL, GOOGL, MSFT)
```

### **Problem: Predictions seem wrong**
```
This is NORMAL!
- Model has ~50% direction accuracy
- Stock market is unpredictable
- Model trained on historical data only
- Not meant for actual trading

Expected behavior: Predictions vary, sometimes wrong
```

### **Problem: Chart not interactive**
```
Solution:
1. Refresh the page (F5)
2. Clear browser cache
3. Try different browser
4. Check JavaScript is enabled
```

---

## ✅ **Testing Checklist**

Use this checklist to verify everything works:

### **Basic Functionality:**
- [ ] App starts successfully
- [ ] Sidebar loads with controls
- [ ] Can enter stock ticker
- [ ] Can select date range
- [ ] Can adjust forecast slider
- [ ] "Generate Prediction" button works

### **Data Fetching:**
- [ ] AAPL data fetches successfully
- [ ] GOOGL data fetches successfully
- [ ] MSFT data fetches successfully
- [ ] Error message for invalid ticker

### **Predictions:**
- [ ] Next-day prediction displays
- [ ] Multi-day forecast displays
- [ ] Predictions are reasonable numbers
- [ ] Direction indicator shows
- [ ] Change percentages calculate correctly

### **Visualization:**
- [ ] Chart renders
- [ ] Historical data shows (blue line)
- [ ] Predictions show (orange dashed)
- [ ] Confidence interval shows (shaded)
- [ ] Chart is interactive (hover works)
- [ ] Can zoom and pan

### **UI Elements:**
- [ ] Metrics cards display
- [ ] Prediction table shows
- [ ] Model info expands
- [ ] Layout is clean and organized
- [ ] No visual glitches

### **Performance:**
- [ ] Predictions generate in <10 seconds
- [ ] UI is responsive
- [ ] No crashes or freezes
- [ ] Multiple predictions work

---

## 🎓 **Understanding Results**

### **What Good Predictions Look Like:**
- Reasonable price range (not 10x current price)
- Small daily changes (typically <5%)
- Confidence intervals widen over time
- Direction changes occasionally

### **What's Normal:**
- ~50% direction accuracy (like coin flip)
- Predictions sometimes wrong
- Wider confidence for longer forecasts
- Variation between stocks

### **What's NOT Normal:**
- Predictions of $0 or negative
- Predictions 10x current price
- No data fetched
- App crashes
- Blank charts

---

## 📝 **Test Report Template**

Use this to document your testing:

```
STOCK PREDICTION APP - TEST REPORT
Date: [Date]
Tester: [Your Name]

TEST 1: AAPL Prediction
Status: [ ] PASS [ ] FAIL
Notes: _______________________

TEST 2: GOOGL Prediction
Status: [ ] PASS [ ] FAIL
Notes: _______________________

TEST 3: Long Forecast (10 days)
Status: [ ] PASS [ ] FAIL
Notes: _______________________

TEST 4: Chart Interaction
Status: [ ] PASS [ ] FAIL
Notes: _______________________

TEST 5: Invalid Ticker
Status: [ ] PASS [ ] FAIL
Notes: _______________________

OVERALL STATUS: [ ] ALL PASS [ ] SOME FAIL
Comments: _______________________
```

---

## 🎯 **Quick Test Commands**

### **Start the app:**
```bash
cd deployment
py -m streamlit run app.py
```

### **Run tests:**
```bash
py test_deployment.py
```

### **Stop the app:**
```
Press Ctrl+C in terminal
```

### **Restart the app:**
```bash
Ctrl+C
py -m streamlit run app.py
```

---

## 🎉 **Success Criteria**

Your app is working correctly if:
- ✅ Loads without errors
- ✅ Fetches data for valid tickers
- ✅ Generates predictions in <10 seconds
- ✅ Charts are interactive
- ✅ Predictions are reasonable
- ✅ UI is clean and responsive
- ✅ Can test multiple stocks
- ✅ No crashes

---

## 📸 **What You Should See**

### **Initial Screen:**
- Clean interface
- Sidebar on left
- Welcome message in center
- Example chart placeholder

### **After Prediction:**
- 3 metric cards at top
- 2 prediction cards
- Large interactive chart
- Detailed table below
- Model info at bottom

### **Chart Features:**
- Blue line (historical)
- Orange dashed (predictions)
- Gray shaded (confidence)
- Hover tooltips
- Zoom controls

---

## 🆘 **Need Help?**

If something doesn't work:
1. Check the terminal for error messages
2. Refresh the browser (F5)
3. Restart the app (Ctrl+C, then restart)
4. Check internet connection
5. Try a different stock ticker
6. Review error messages carefully

---

## 🎊 **You're Ready!**

Your app is fully functional and ready to test. Have fun predicting stock prices!

**Remember:** This is for educational purposes only. Don't use for actual trading! 📚

---

**Happy Testing! 🚀📈**
