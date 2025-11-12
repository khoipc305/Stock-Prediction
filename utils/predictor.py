"""
Stock price prediction using trained LSTM model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from src.models_lstm import PriceLSTM, EarlyFusionLSTM
except ImportError:
    # Fallback: define models here if import fails
    class PriceLSTM(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
            super(PriceLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.dropout(lstm_out[:, -1, :])
            return self.fc(out)
    
    class EarlyFusionLSTM(nn.Module):
        def __init__(self, price_features, sentiment_features, hidden_size=64, num_layers=2, dropout=0.2, output_size=1):
            super(EarlyFusionLSTM, self).__init__()
            input_size = price_features + sentiment_features
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.dropout(lstm_out[:, -1, :])
            return self.fc(out)

from .preprocessor import prepare_features, create_sequences
from sklearn.preprocessing import StandardScaler

class StockPredictor:
    """
    Stock price predictor using trained LSTM model
    """
    
    def __init__(self, model_path, lookback=30):
        """
        Initialize predictor
        
        Parameters:
        -----------
        model_path : str
            Path to trained model checkpoint
        lookback : int
            Number of time steps for LSTM input
        """
        self.lookback = lookback
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model, self.scaler, self.feature_list = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load trained model and scaler"""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model configuration from checkpoint
        # Handle both old and new checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
            input_size = config['input_size']
            hidden_size = config['hidden_size']
            num_layers = config['num_layers']
            dropout = config.get('dropout', 0.2)
            model_type = config.get('model_type', 'early_fusion')
        else:
            # Fallback for old format
            input_size = checkpoint.get('input_size', 25)
            hidden_size = checkpoint.get('hidden_size', 64)
            num_layers = checkpoint.get('num_layers', 2)
            dropout = checkpoint.get('dropout', 0.2)
            model_type = checkpoint.get('model_type', 'early_fusion')
        
        # Initialize model based on type
        if model_type == 'price_lstm':
            model = PriceLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        else:  # early_fusion or default
            # For early fusion, we need to know price vs sentiment features
            # Assume last 8 features are sentiment (based on training)
            price_features = input_size - 8
            sentiment_features = 8
            model = EarlyFusionLSTM(
                price_features=price_features,
                sentiment_features=sentiment_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Load scaler (reconstruct from saved params)
        scaler = StandardScaler()
        if 'scaler_params' in checkpoint:
            import numpy as np
            params = checkpoint['scaler_params']
            scaler.mean_ = np.array(params['mean'])
            # Handle both 'scale' and 'std' keys
            if 'scale' in params:
                scaler.scale_ = np.array(params['scale'])
            elif 'std' in params:
                scaler.scale_ = np.array(params['std'])
            # Calculate variance from std if needed
            if 'var' in params:
                scaler.var_ = np.array(params['var'])
            else:
                scaler.var_ = scaler.scale_ ** 2
            scaler.n_features_in_ = len(scaler.mean_)
        
        # Load feature list
        feature_list = checkpoint.get('feature_cols', None)
        
        return model, scaler, feature_list
    
    def predict(self, data, forecast_days=5):
        """
        Make predictions for future prices
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical stock data
        forecast_days : int
            Number of days to forecast
        
        Returns:
        --------
        dict
            Predictions with prices and confidence intervals
        """
        # Prepare features
        features = prepare_features(data, self.feature_list)
        
        if len(features) < self.lookback:
            raise ValueError(f"Need at least {self.lookback} days of data")
        
        # Get current price (last close price)
        current_price = data['Close'].iloc[-1]
        
        # Scale features
        scaled_features = self.scaler.transform(features.values)
        
        # Get last sequence
        last_sequence = scaled_features[-self.lookback:]
        
        # Make predictions (model predicts RETURNS, not prices)
        predicted_returns = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(forecast_days):
                # Prepare input
                x = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Predict return (not price!)
                pred_return = self.model(x).cpu().numpy()[0, 0]
                predicted_returns.append(pred_return)
                
                # Update sequence for next prediction
                # Create new row with updated features
                new_row = current_sequence[-1].copy()
                
                # Update the return feature (typically at index 5 for return_1d)
                # But keep it scaled
                new_row[5] = pred_return  # Scaled return prediction
                
                current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Convert predicted returns to prices
        # The model predicts returns, so we need to apply them to prices
        predicted_prices = []
        price = current_price
        
        # Calculate historical volatility for confidence intervals
        hist_returns = data['Close'].pct_change().dropna()
        hist_std = hist_returns.std()
        
        for pred_return in predicted_returns:
            # Apply the predicted return to get next price
            # pred_return is scaled, so we need to unscale it first
            
            # Create dummy array for inverse transform
            dummy = np.zeros(scaled_features.shape[1])
            dummy[5] = pred_return  # return_1d is at index 5
            
            # Unscale the return
            unscaled = self.scaler.inverse_transform(dummy.reshape(1, -1))[0]
            actual_return = unscaled[5]
            
            # Clip extreme predictions (safety measure)
            actual_return = np.clip(actual_return, -0.1, 0.1)  # Max ±10% per day
            
            # Apply return to price
            price = price * (1 + actual_return)
            predicted_prices.append(price)
        
        # Calculate confidence intervals based on historical volatility
        confidence_lower = []
        confidence_upper = []
        price = current_price
        
        for i in range(forecast_days):
            # Confidence grows with forecast horizon
            std_multiplier = np.sqrt(i + 1)  # Sqrt of days ahead
            lower = price * (1 - 1.96 * hist_std * std_multiplier)
            upper = price * (1 + 1.96 * hist_std * std_multiplier)
            confidence_lower.append(lower)
            confidence_upper.append(upper)
            price = predicted_prices[i]
        
        return {
            'prices': predicted_prices,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper
        }
    
    def predict_direction(self, data):
        """
        Predict if price will go up or down
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical stock data
        
        Returns:
        --------
        dict
            Direction prediction and confidence
        """
        prediction = self.predict(data, forecast_days=1)
        current_price = data['Close'].iloc[-1]
        predicted_price = prediction['prices'][0]
        
        direction = "UP" if predicted_price > current_price else "DOWN"
        change_pct = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'direction': direction,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_percent': change_pct
        }
