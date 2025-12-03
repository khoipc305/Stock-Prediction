"""
Evaluate trained models and run backtests.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional
import json

from .utils import (
    create_sequences,
    split_by_date,
    normalize_features,
    calculate_metrics,
    print_metrics
)
from .models_lstm import create_model


def load_model(model_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    model = create_model(
        model_type=config['model_type'],
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        output_size=1
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint['epoch']} epochs)")
    
    return model, checkpoint


def predict(model: torch.nn.Module,
           sequences: np.ndarray,
           device: torch.device,
           batch_size: int = 64) -> np.ndarray:
    """
    Make predictions on sequences.
    
    Args:
        model: Trained model
        sequences: Input sequences
        device: Device
        batch_size: Batch size
    
    Returns:
        Predictions array
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            preds = model(batch_tensor)
            predictions.append(preds.cpu().numpy())
    
    return np.concatenate(predictions).flatten()


def evaluate_model(dataset_path: str,
                  model_path: str,
                  output_dir: str = 'reports/figures',
                  target_col: str = 'target_return') -> dict:
    """
    Evaluate model on test set and generate plots.
    
    Args:
        dataset_path: Path to dataset
        model_path: Path to trained model
        output_dir: Directory to save plots
        target_col: Target column name
    
    Returns:
        Dictionary with evaluation results
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    config = checkpoint['config']
    scaler_params = checkpoint['scaler_params']
    feature_cols = checkpoint['feature_cols']
    lookback = config['lookback']
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Split data
    train_df, val_df, test_df = split_by_date(df)
    
    # Prepare test data
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Normalize
    mean = scaler_params['mean']
    std = scaler_params['std']
    X_test_norm = (X_test - mean) / std
    
    # Create sequences
    X_test_seq, y_test_seq = create_sequences(X_test_norm, lookback, y_test)
    
    print(f"Test sequences: {X_test_seq.shape}")
    
    # Predict
    print("\nMaking predictions...")
    predictions = predict(model, X_test_seq, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_seq, predictions)
    print_metrics(metrics, prefix="Test Set")
    
    # Baseline: naive (yesterday's return)
    naive_preds = np.zeros_like(y_test_seq)
    naive_metrics = calculate_metrics(y_test_seq, naive_preds)
    print_metrics(naive_metrics, prefix="Naive Baseline")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Predictions vs Actual
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test_seq, predictions, alpha=0.5, s=10)
    plt.plot([y_test_seq.min(), y_test_seq.max()], 
             [y_test_seq.min(), y_test_seq.max()], 
             'r--', lw=2, label='Perfect prediction')
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Predictions vs Actual Returns (Test Set)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_vs_actual.png', dpi=150)
    print(f"\nSaved plot: {output_dir / 'predictions_vs_actual.png'}")
    plt.close()
    
    # Plot 2: Time series of predictions
    test_dates = test_df.index[lookback:]
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_seq, label='Actual', alpha=0.7, linewidth=1)
    plt.plot(test_dates, predictions, label='Predicted', alpha=0.7, linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.title('Predicted vs Actual Returns Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_predictions.png', dpi=150)
    print(f"Saved plot: {output_dir / 'time_series_predictions.png'}")
    plt.close()
    
    # Plot 3: Residuals
    residuals = y_test_seq - predictions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(predictions, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Returns')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'residuals.png', dpi=150)
    print(f"Saved plot: {output_dir / 'residuals.png'}")
    plt.close()
    
    # Save results
    results = {
        'model_metrics': metrics,
        'baseline_metrics': naive_metrics,
        'test_samples': len(y_test_seq),
        'config': config
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def simple_backtest(dataset_path: str,
                   model_path: str,
                   strategy: str = 'long',
                   transaction_cost: float = 0.001,
                   output_dir: str = 'reports/figures') -> dict:
    """
    Run simple backtest on test set.
    
    Args:
        dataset_path: Path to dataset
        model_path: Path to trained model
        strategy: 'long' (buy positive predictions) or 'long_short'
        transaction_cost: Transaction cost (e.g., 0.001 = 10 bps)
        output_dir: Directory to save plots
    
    Returns:
        Dictionary with backtest results
    """
    print("\n" + "="*60)
    print("RUNNING BACKTEST")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    config = checkpoint['config']
    scaler_params = checkpoint['scaler_params']
    feature_cols = checkpoint['feature_cols']
    lookback = config['lookback']
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Split data
    train_df, val_df, test_df = split_by_date(df)
    
    # Prepare test data
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_return'].values
    
    # Normalize
    mean = scaler_params['mean']
    std = scaler_params['std']
    X_test_norm = (X_test - mean) / std
    
    # Create sequences
    X_test_seq, y_test_seq = create_sequences(X_test_norm, lookback, y_test)
    
    # Predict
    predictions = predict(model, X_test_seq, device)
    
    # Backtest
    test_dates = test_df.index[lookback:]
    
    if strategy == 'long':
        # Long only: buy when prediction > 0
        positions = (predictions > 0).astype(float)
    elif strategy == 'long_short':
        # Long-short: long when pred > 0, short when pred < 0
        positions = np.sign(predictions)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Calculate returns
    position_changes = np.abs(np.diff(positions, prepend=0))
    costs = position_changes * transaction_cost
    
    strategy_returns = positions * y_test_seq - costs
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Buy and hold benchmark
    buy_hold_returns = (1 + y_test_seq).cumprod()
    
    # Metrics
    total_return = cumulative_returns[-1] - 1
    buy_hold_return = buy_hold_returns[-1] - 1
    
    sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
    max_drawdown = np.min(cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1)
    
    win_rate = np.mean(strategy_returns > 0)
    
    print(f"\nBacktest Results ({strategy} strategy):")
    print(f"  Total Return:     {total_return:.2%}")
    print(f"  Buy & Hold:       {buy_hold_return:.2%}")
    print(f"  Sharpe Ratio:     {sharpe_ratio:.2f}")
    print(f"  Max Drawdown:     {max_drawdown:.2%}")
    print(f"  Win Rate:         {win_rate:.2%}")
    print(f"  Avg Daily Return: {np.mean(strategy_returns):.4%}")
    
    # Plot equity curve
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, cumulative_returns, label=f'{strategy.title()} Strategy', linewidth=2)
    plt.plot(test_dates, buy_hold_returns, label='Buy & Hold', linewidth=2, alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.title(f'Backtest: {strategy.title()} Strategy vs Buy & Hold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'backtest_equity_curve.png', dpi=150)
    print(f"\nSaved plot: {output_dir / 'backtest_equity_curve.png'}")
    plt.close()
    
    # Save results
    results = {
        'strategy': strategy,
        'total_return': float(total_return),
        'buy_hold_return': float(buy_hold_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'avg_daily_return': float(np.mean(strategy_returns)),
        'transaction_cost': transaction_cost
    }
    
    results_path = output_dir / 'backtest_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    return results


def main():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained model and run backtest')
    parser.add_argument('--dataset', required=True,
                       help='Path to dataset parquet')
    parser.add_argument('--weights', required=True,
                       help='Path to trained model weights')
    parser.add_argument('--strategy', default='long',
                       choices=['long', 'long_short'],
                       help='Backtest strategy')
    parser.add_argument('--cost', type=float, default=0.001,
                       help='Transaction cost (default: 0.001 = 10 bps)')
    parser.add_argument('--output', default='reports/figures',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Evaluate model
    evaluate_model(
        dataset_path=args.dataset,
        model_path=args.weights,
        output_dir=args.output
    )
    
    # Run backtest
    simple_backtest(
        dataset_path=args.dataset,
        model_path=args.weights,
        strategy=args.strategy,
        transaction_cost=args.cost,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
