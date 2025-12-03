"""
PyTorch LSTM models for stock price prediction.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PriceLSTM(nn.Module):
    """
    Basic LSTM for price prediction using only price features.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of outputs (1 for regression)
        """
        super(PriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Dropout and fully connected
        out = self.dropout(last_hidden)
        out = self.fc(out)
        
        return out


class EarlyFusionLSTM(nn.Module):
    """
    LSTM with early fusion - sentiment features concatenated at each timestep.
    """
    
    def __init__(self,
                 price_features: int,
                 sentiment_features: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        """
        Args:
            price_features: Number of price features
            sentiment_features: Number of sentiment features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size: Number of outputs
        """
        super(EarlyFusionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM takes combined features
        input_size = price_features + sentiment_features
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, price_features + sentiment_features)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out


class LateFusionLSTM(nn.Module):
    """
    Late fusion - separate LSTM for prices and MLP for sentiment, then concatenate.
    """
    
    def __init__(self,
                 price_features: int,
                 sentiment_features: int,
                 lstm_hidden: int = 64,
                 lstm_layers: int = 2,
                 mlp_hidden: int = 32,
                 dropout: float = 0.2,
                 output_size: int = 1):
        """
        Args:
            price_features: Number of price features
            sentiment_features: Number of sentiment features
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            mlp_hidden: MLP hidden size for sentiment
            dropout: Dropout rate
            output_size: Number of outputs
        """
        super(LateFusionLSTM, self).__init__()
        
        # Price LSTM
        self.price_lstm = nn.LSTM(
            input_size=price_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Sentiment MLP (processes only last timestep)
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(sentiment_features, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU()
        )
        
        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden + mlp_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, price_seq: torch.Tensor, sentiment_last: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            price_seq: Price sequence (batch_size, seq_len, price_features)
            sentiment_last: Sentiment at last timestep (batch_size, sentiment_features)
        
        Returns:
            Output tensor (batch_size, output_size)
        """
        # Process price sequence
        lstm_out, _ = self.price_lstm(price_seq)
        price_repr = lstm_out[:, -1, :]
        
        # Process sentiment
        sentiment_repr = self.sentiment_mlp(sentiment_last)
        
        # Concatenate and fuse
        combined = torch.cat([price_repr, sentiment_repr], dim=1)
        out = self.fusion_head(combined)
        
        return out


class DirectionClassifier(nn.Module):
    """
    LSTM-based classifier for predicting direction (up/down/neutral).
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 num_classes: int = 3):
        """
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            num_classes: Number of classes (3 for up/down/neutral)
        """
        super(DirectionClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
        
        Returns:
            Logits (batch_size, num_classes)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


def create_model(model_type: str,
                input_size: int,
                sentiment_size: int = 0,
                hidden_size: int = 64,
                num_layers: int = 2,
                dropout: float = 0.2,
                output_size: int = 1) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('price_lstm', 'early_fusion', 'late_fusion', 'classifier')
        input_size: Number of input features (price features for fusion models)
        sentiment_size: Number of sentiment features (for fusion models)
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        output_size: Number of outputs
    
    Returns:
        PyTorch model
    """
    if model_type == 'price_lstm':
        return PriceLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    
    elif model_type == 'early_fusion':
        return EarlyFusionLSTM(
            price_features=input_size,
            sentiment_features=sentiment_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    
    elif model_type == 'late_fusion':
        return LateFusionLSTM(
            price_features=input_size,
            sentiment_features=sentiment_size,
            lstm_hidden=hidden_size,
            lstm_layers=num_layers,
            dropout=dropout,
            output_size=output_size
        )
    
    elif model_type == 'classifier':
        return DirectionClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=output_size
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    batch_size = 32
    seq_len = 30
    price_features = 16
    sentiment_features = 8
    
    print("Testing PriceLSTM...")
    model = PriceLSTM(input_size=price_features)
    x = torch.randn(batch_size, seq_len, price_features)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\nTesting EarlyFusionLSTM...")
    model = EarlyFusionLSTM(price_features=price_features, sentiment_features=sentiment_features)
    x = torch.randn(batch_size, seq_len, price_features + sentiment_features)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\nTesting LateFusionLSTM...")
    model = LateFusionLSTM(price_features=price_features, sentiment_features=sentiment_features)
    price_seq = torch.randn(batch_size, seq_len, price_features)
    sentiment_last = torch.randn(batch_size, sentiment_features)
    out = model(price_seq, sentiment_last)
    print(f"  Price: {price_seq.shape}, Sentiment: {sentiment_last.shape}, Output: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    print("\nTesting DirectionClassifier...")
    model = DirectionClassifier(input_size=price_features, num_classes=3)
    x = torch.randn(batch_size, seq_len, price_features)
    out = model(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
