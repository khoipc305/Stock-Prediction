"""
Setup script for deployment
"""

import shutil
from pathlib import Path

def setup_deployment():
    """
    Setup deployment environment
    """
    print("🚀 Setting up Stock Prediction Deployment...")
    print()
    
    # Create directories
    dirs = ['models', 'data', 'static', 'templates']
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}/")
    
    print()
    
    # Copy model file
    model_source = Path('../models/lstm_early_fusion.pt')
    model_dest = Path('models/lstm_early_fusion.pt')
    
    if model_source.exists():
        shutil.copy(model_source, model_dest)
        print(f"✓ Copied model: {model_source} -> {model_dest}")
    else:
        print(f"⚠️  Model not found: {model_source}")
        print("   Please train your model first by running notebook 03_train_lstm.ipynb")
    
    print()
    
    # Check requirements
    print("📦 Checking requirements...")
    try:
        import streamlit
        print("✓ streamlit installed")
    except ImportError:
        print("✗ streamlit not installed - run: pip install -r requirements.txt")
    
    try:
        import yfinance
        print("✓ yfinance installed")
    except ImportError:
        print("✗ yfinance not installed - run: pip install -r requirements.txt")
    
    try:
        import plotly
        print("✓ plotly installed")
    except ImportError:
        print("✗ plotly not installed - run: pip install -r requirements.txt")
    
    print()
    print("=" * 60)
    print("✅ Setup complete!")
    print()
    print("To run the application:")
    print("  streamlit run app.py")
    print()
    print("=" * 60)

if __name__ == "__main__":
    setup_deployment()
