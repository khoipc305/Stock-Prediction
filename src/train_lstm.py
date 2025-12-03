"""
Train LSTM models for stock prediction.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional
import json
from tqdm import tqdm

from .utils import (
    set_seed,
    create_sequences,
    split_by_date,
    normalize_features,
    EarlyStopping,
    calculate_metrics,
    print_metrics
)
from .models_lstm import create_model, count_parameters


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: Feature sequences (n_samples, lookback, n_features)
            targets: Target values (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_and_prepare_data(dataset_path: str,
                          lookback: int = 30,
                          train_end: str = '2021-12-31',
                          val_end: str = '2023-12-31',
                          target_col: str = 'target_return') -> Tuple:
    """
    Load dataset and prepare sequences.
    
    Args:
        dataset_path: Path to dataset parquet
        lookback: Number of timesteps to look back
        train_end: End date for training
        val_end: End date for validation
        target_col: Target column name
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler_params, feature_cols)
    """
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Get feature columns (exclude metadata and targets)
    exclude_cols = ['ticker', 'target_close', 'target_return', 'target_direction']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col}")
    
    # Split by date
    train_df, val_df, test_df = split_by_date(df, train_end=train_end, val_end=val_end)
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val:   {len(val_df)} rows")
    print(f"  Test:  {len(test_df)} rows")
    
    # Extract features and targets
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Normalize features
    X_train_norm, X_val_norm, X_test_norm, scaler_params = normalize_features(
        X_train, X_val, X_test
    )
    
    # Create sequences
    print(f"\nCreating sequences with lookback={lookback}...")
    X_train_seq, y_train_seq = create_sequences(X_train_norm, lookback, y_train)
    X_val_seq, y_val_seq = create_sequences(X_val_norm, lookback, y_val)
    X_test_seq, y_test_seq = create_sequences(X_test_norm, lookback, y_test)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Val sequences:   {X_val_seq.shape}")
    print(f"  Test sequences:  {X_test_seq.shape}")
    
    # Create datasets and loaders
    train_dataset = StockDataset(X_train_seq, y_train_seq)
    val_dataset = StockDataset(X_val_seq, y_val_seq)
    test_dataset = StockDataset(X_test_seq, y_test_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler_params, feature_cols


def train_epoch(model: nn.Module,
               loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss
    """
    model.train()
    total_loss = 0
    
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device).unsqueeze(1)
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model: nn.Module,
            loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model.
    
    Args:
        model: PyTorch model
        loader: DataLoader
        criterion: Loss function
        device: Device
    
    Returns:
        Tuple of (loss, predictions, targets)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    predictions = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()
    
    return avg_loss, predictions, targets


def train_model(dataset_path: str,
               lookback: int = 30,
               hidden_size: int = 64,
               num_layers: int = 2,
               dropout: float = 0.2,
               learning_rate: float = 0.001,
               epochs: int = 60,
               patience: int = 8,
               model_type: str = 'price_lstm',
               save_path: str = 'models/best_lstm.pt',
               seed: int = 42) -> dict:
    """
    Train LSTM model.
    
    Args:
        dataset_path: Path to dataset
        lookback: Sequence length
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        learning_rate: Learning rate
        epochs: Maximum epochs
        patience: Early stopping patience
        model_type: Type of model
        save_path: Path to save best model
        seed: Random seed
    
    Returns:
        Dictionary with training history
    """
    # Set seed
    set_seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader, scaler_params, feature_cols = load_and_prepare_data(
        dataset_path=dataset_path,
        lookback=lookback
    )
    
    input_size = len(feature_cols)
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = create_model(
        model_type=model_type,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1
    )
    model = model.to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_direction_acc': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\nTraining for up to {epochs} epochs...")
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        
        # Calculate metrics
        val_metrics = calculate_metrics(val_targets, val_preds)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_direction_acc'].append(val_metrics['direction_accuracy'])
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val MAE:    {val_metrics['mae']:.6f}")
            print(f"  Val Dir Acc: {val_metrics['direction_accuracy']:.2%}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'scaler_params': scaler_params,
                'feature_cols': feature_cols,
                'config': {
                    'model_type': model_type,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'lookback': lookback
                }
            }, save_path)
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Model saved to {save_path}")
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    test_metrics = calculate_metrics(test_targets, test_preds)
    
    print_metrics(test_metrics, prefix="Test")
    
    # Save training history (convert numpy types to Python types for JSON)
    def convert_to_python_types(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    history_path = save_path.parent / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(convert_to_python_types(history), f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    return history


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train LSTM model for stock prediction')
    parser.add_argument('--dataset', required=True,
                       help='Path to dataset parquet')
    parser.add_argument('--lookback', type=int, default=30,
                       help='Sequence length')
    parser.add_argument('--hidden', type=int, default=64,
                       help='LSTM hidden size')
    parser.add_argument('--layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=60,
                       help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=8,
                       help='Early stopping patience')
    parser.add_argument('--model_type', default='price_lstm',
                       choices=['price_lstm', 'early_fusion', 'late_fusion'],
                       help='Model architecture')
    parser.add_argument('--save', default='models/best_lstm.pt',
                       help='Path to save model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset,
        lookback=args.lookback,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        model_type=args.model_type,
        save_path=args.save,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
