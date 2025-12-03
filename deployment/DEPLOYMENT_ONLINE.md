# 🌐 Deploy Your App Online

## Get a Public URL Like: `https://stock-predictor.streamlit.app`

---

## 🚀 **Option 1: Streamlit Community Cloud (FREE & EASIEST)**

### **What You Get:**
- ✅ Free hosting forever
- ✅ Public URL: `https://your-app.streamlit.app`
- ✅ Automatic updates when you push to GitHub
- ✅ HTTPS (secure)
- ✅ No credit card required

---

### **📋 Step-by-Step Guide:**

#### **Step 1: Create a GitHub Repository**

1. **Go to GitHub:** https://github.com
2. **Sign in** (or create account if needed)
3. **Click:** "New repository" button
4. **Fill in:**
   - Repository name: `stock-prediction-app`
   - Description: `LSTM-based stock price prediction web app`
   - Visibility: **Public** (required for free hosting)
5. **Click:** "Create repository"

---

#### **Step 2: Upload Your Code to GitHub**

**Option A: Using GitHub Web Interface (Easiest)**

1. **In your new repository**, click "uploading an existing file"
2. **Drag and drop these files/folders:**
   ```
   deployment/
   ├── app_simple.py (rename to app.py when uploading)
   ├── requirements.txt
   ├── config.py
   ├── packages.txt
   ├── .streamlit/
   │   └── config.toml
   ├── models/
   │   └── lstm_early_fusion.pt
   ├── utils/
   │   ├── __init__.py
   │   ├── data_fetcher.py
   │   ├── preprocessor.py
   │   └── predictor.py
   └── README.md
   ```
3. **Important:** Rename `app_simple.py` to `app.py` before uploading
4. **Commit changes**

**Option B: Using Git Command Line**

```bash
cd deployment

# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Stock prediction app"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/stock-prediction-app.git

# Push
git branch -M main
git push -u origin main
```

---

#### **Step 3: Deploy on Streamlit Cloud**

1. **Go to:** https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click:** "New app" button
4. **Fill in:**
   - **Repository:** `YOUR_USERNAME/stock-prediction-app`
   - **Branch:** `main`
   - **Main file path:** `app.py` (or `app_simple.py` if you didn't rename)
5. **Click:** "Deploy!"

---

#### **Step 4: Wait for Deployment**

- ⏱️ Takes 2-5 minutes
- 📊 You'll see build logs
- ✅ When done, you get your public URL!

---

### **🎉 Your App is Live!**

You'll get a URL like:
```
https://stock-prediction-app-YOUR_USERNAME.streamlit.app
```

**Share this URL with anyone!** They can:
- Access it from any device
- No installation needed
- Works on mobile too!

---

## 🔧 **Important Files for Deployment**

### **1. requirements.txt**
Already created! Contains:
```
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
torch==2.1.2
yfinance==0.2.33
plotly==5.18.0
scikit-learn==1.3.2
python-dateutil==2.8.2
```

### **2. app.py** (Main file)
Use `app_simple.py` and rename to `app.py`

### **3. models/lstm_early_fusion.pt**
Your trained model (must be included!)

### **4. .streamlit/config.toml**
Already created! Configures the app appearance

---

## 📁 **Repository Structure**

Your GitHub repo should look like:
```
stock-prediction-app/
├── app.py                    # Main application
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages (optional)
├── config.py                 # Configuration
├── README.md                 # Documentation
├── .streamlit/
│   └── config.toml          # Streamlit config
├── models/
│   └── lstm_early_fusion.pt # Trained model
└── utils/
    ├── __init__.py
    ├── data_fetcher.py
    ├── preprocessor.py
    └── predictor.py
```

---

## ⚠️ **Important Notes**

### **Model File Size:**
- Your model is ~450 KB ✅ (perfect!)
- GitHub limit: 100 MB
- Streamlit Cloud limit: 1 GB

### **Free Tier Limits:**
- ✅ Unlimited apps
- ✅ Unlimited visitors
- ⚠️ 1 GB RAM per app
- ⚠️ 1 CPU core per app
- ⚠️ Apps sleep after 7 days of inactivity

### **Privacy:**
- Repository must be **public** for free hosting
- Anyone can see your code
- Your model file will be public

---

## 🎨 **Customize Your URL**

After deployment, you can:
1. Go to app settings
2. Change the app name
3. Get a custom URL like:
   ```
   https://stock-predictor-yourname.streamlit.app
   ```

---

## 🔄 **Update Your App**

To update your deployed app:
1. Make changes locally
2. Push to GitHub:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push
   ```
3. Streamlit Cloud auto-deploys! ✨

---

## 🐛 **Troubleshooting Deployment**

### **Problem: Build fails**
**Check:**
- All files uploaded correctly?
- `requirements.txt` has correct versions?
- Model file included?

### **Problem: App crashes**
**Check:**
- Logs in Streamlit Cloud dashboard
- Memory usage (free tier: 1 GB)
- File paths are correct

### **Problem: Model not found**
**Solution:**
- Ensure `models/lstm_early_fusion.pt` is in repo
- Check path in `predictor.py`: `'models/lstm_early_fusion.pt'`

---

## 🌟 **Alternative Deployment Options**

### **Option 2: Heroku (Free tier discontinued)**
- More complex setup
- Need Procfile and setup.sh
- Not recommended for beginners

### **Option 3: AWS/Azure/GCP**
- Professional deployment
- Costs money
- More control and scalability

### **Option 4: Share Locally (Network)**
If you just want to share on your local network:

```bash
# Find your IP address
ipconfig  # Windows

# Run with network access
py -m streamlit run app_simple.py --server.address 0.0.0.0

# Share this URL with others on your network:
http://YOUR_IP_ADDRESS:8501
```

---

## 📊 **Example: Successful Deployment**

**Your app URL might look like:**
```
https://stock-prediction-calpolypomona.streamlit.app
```

**Features:**
- ✅ Anyone can access
- ✅ Works on mobile
- ✅ HTTPS secure
- ✅ Fast loading
- ✅ Auto-updates

---

## 🎓 **Quick Checklist**

Before deploying, ensure:
- [ ] GitHub account created
- [ ] Repository created (public)
- [ ] All files uploaded
- [ ] `app.py` is the main file
- [ ] `requirements.txt` included
- [ ] Model file uploaded
- [ ] Streamlit Cloud account created
- [ ] App deployed
- [ ] URL works
- [ ] Tested with different stocks

---

## 📝 **Sample README for GitHub**

Create this as your repository's `README.md`:

```markdown
# 📈 Stock Price Prediction App

LSTM-based stock price prediction using deep learning and sentiment analysis.

## 🚀 Live Demo
[Try it here!](https://your-app-url.streamlit.app)

## 🎯 Features
- Real-time stock data fetching
- LSTM predictions (1-30 days)
- Interactive charts
- Technical indicators
- Confidence intervals

## 🛠️ Tech Stack
- Python 3.12
- Streamlit
- PyTorch
- YFinance
- Plotly

## 📊 Model
- Architecture: LSTM (Early Fusion)
- Features: 25 (price + technical + sentiment)
- Training: 2015-2021 AAPL data
- Accuracy: ~50% direction

## ⚠️ Disclaimer
Educational purposes only. Not financial advice.

## 👨‍🎓 Author
CS4200 - Cal Poly Pomona
```

---

## 🎉 **You're Ready to Deploy!**

Follow the steps above and your app will be live in minutes!

**Questions?**
- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io

---

**Good luck with your deployment! 🚀**
