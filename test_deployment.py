"""
Test script to verify deployment setup
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages are installed"""
    print("Testing imports...")
    errors = []
    
    try:
        import streamlit
        print("✓ streamlit")
    except ImportError as e:
        errors.append(f"✗ streamlit: {e}")
    
    try:
        import pandas
        print("✓ pandas")
    except ImportError as e:
        errors.append(f"✗ pandas: {e}")
    
    try:
        import numpy
        print("✓ numpy")
    except ImportError as e:
        errors.append(f"✗ numpy: {e}")
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        errors.append(f"✗ torch: {e}")
    
    try:
        import yfinance
        print("✓ yfinance")
    except ImportError as e:
        errors.append(f"✗ yfinance: {e}")
    
    try:
        import plotly
        print("✓ plotly")
    except ImportError as e:
        errors.append(f"✗ plotly: {e}")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        errors.append(f"✗ scikit-learn: {e}")
    
    return errors

def test_files():
    """Test if all required files exist"""
    print("\nTesting files...")
    errors = []
    
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'utils/__init__.py',
        'utils/data_fetcher.py',
        'utils/preprocessor.py',
        'utils/predictor.py'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            errors.append(f"✗ {file} not found")
    
    return errors

def test_model():
    """Test if model file exists"""
    print("\nTesting model...")
    model_path = Path('models/lstm_early_fusion.pt')
    
    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        return []
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("   Run: python setup.py")
        print("   Or copy manually from ../models/")
        return ["Model file missing"]

def test_data_fetch():
    """Test if we can fetch stock data"""
    print("\nTesting data fetching...")
    try:
        from utils.data_fetcher import fetch_stock_data
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        data = fetch_stock_data('AAPL', start_date, end_date)
        
        if data is not None and len(data) > 0:
            print(f"✓ Successfully fetched {len(data)} days of AAPL data")
            return []
        else:
            return ["Could not fetch stock data"]
    except Exception as e:
        return [f"Data fetch error: {e}"]

def main():
    """Run all tests"""
    print("=" * 60)
    print("DEPLOYMENT TEST SUITE")
    print("=" * 60)
    print()
    
    all_errors = []
    
    # Test imports
    errors = test_imports()
    all_errors.extend(errors)
    
    # Test files
    errors = test_files()
    all_errors.extend(errors)
    
    # Test model
    errors = test_model()
    all_errors.extend(errors)
    
    # Test data fetching
    errors = test_data_fetch()
    all_errors.extend(errors)
    
    # Summary
    print()
    print("=" * 60)
    if not all_errors:
        print("✅ ALL TESTS PASSED!")
        print()
        print("You're ready to run the application:")
        print("  streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED:")
        for error in all_errors:
            print(f"  - {error}")
        print()
        print("Please fix the errors above before running the app.")
        print()
        print("Common fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Copy model file: python setup.py")
        print("  3. Check internet connection")
    print("=" * 60)

if __name__ == "__main__":
    main()
