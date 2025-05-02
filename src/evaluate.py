"""
This module provides functionality for evaluating trained GRU forecasting models.
It includes utilities for loading checkpoints, making predictions, computing metrics,
and generating visualizations of the results.
"""

import argparse
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict

import joblib
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_series
from models import GRUForecast

# -------- Configuration --------

@dataclass
class EvalConfig:
    """
    Configuration class for model evaluation.
    
    This dataclass holds all the necessary parameters for running the evaluation:
    - Paths to model checkpoint, scaler, and data files
    - Number of steps to forecast
    - Output directory for results
    - Logging level
    """
    checkpoint: Path  # Path to the model checkpoint file
    scaler: Optional[Path]  # Path to the scaler file (optional)
    hist_data: Path  # Path to historical data file
    test_data: Optional[Path]  # Path to test data file (optional)
    n_steps: int  # Number of steps to forecast
    results_dir: Path  # Directory to save evaluation results
    log_level: int = logging.INFO  # Logging level

# -------- Logger Setup --------

def setup_logger(level: int) -> logging.Logger:
    """
    Configure and return a logger with the specified level.
    
    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    return logging.getLogger(__name__)

# -------- I/O Utilities --------
def ensure_dir(path: Path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)

# -------- Checkpoint Loading --------
def load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict, torch.device]:
    """
    Load a model checkpoint from disk.
    
    Args:
        path: Path to the checkpoint file
        device: Device to load the checkpoint onto
        
    Returns:
        Tuple[Dict, torch.device]: 
            - Checkpoint dictionary containing model state and hyperparameters
            - Device the checkpoint was loaded onto
            
    Raises:
        FileNotFoundError: If the checkpoint file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logging.info(f"Loading checkpoint from {path} on {device}")
    ckpt = torch.load(path, map_location=device)
    return ckpt, device

# -------- Model Instantiation --------
def instantiate_model(ckpt: Dict, device: torch.device) -> Tuple[nn.Module, int]:
    """
    Create and load a model from a checkpoint.
    
    Args:
        ckpt: Checkpoint dictionary containing model state and hyperparameters
        device: Device to load the model onto
        
    Returns:
        Tuple[nn.Module, int]:
            - Loaded and configured GRU model
            - Lookback window size used by the model
            
    Raises:
        ValueError: If checkpoint is missing required parameters
    """
    params = ckpt.get('hyperparams')
    state = ckpt.get('model_state')
    if params is None or state is None:
        raise ValueError("Checkpoint missing 'hyperparams' or 'model_state'")
    lookback = params.get('lookback')
    if lookback is None:
        raise ValueError("Hyperparameters missing 'lookback'")

    model = GRUForecast(
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params.get('dropout', 0.0),
        lookback=lookback
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    logging.info("Model loaded and ready")
    return model, lookback

# -------- Forecasting --------
def recursive_forecast(
    model: nn.Module,
    init_seq: np.ndarray,
    lookback: int,
    n_steps: int,
    device: torch.device
) -> np.ndarray:
    """
    Generate a multi-step forecast using recursive prediction.
    
    This function uses the model to predict one step at a time, using each
    prediction as input for the next step. This is known as recursive or
    autoregressive forecasting.
    
    Args:
        model: Trained GRU model
        init_seq: Initial sequence of normalized values
        lookback: Number of previous steps used for prediction
        n_steps: Number of steps to forecast
        device: Device to run the model on
        
    Returns:
        np.ndarray: Array of predicted values
        
    Raises:
        ValueError: If initial sequence is shorter than lookback window
    """
    if init_seq.shape[0] < lookback:
        raise ValueError(f"Initial sequence length {init_seq.shape[0]} < lookback {lookback}")
    window = init_seq[-lookback:].tolist()
    preds = []
    with torch.no_grad():
        for _ in range(n_steps):
            tensor = torch.tensor(window[-lookback:], dtype=torch.float32)
            tensor = tensor.view(1, lookback, 1).to(device)
            out = model(tensor).cpu().item()
            preds.append(out)
            window.append(out)
    return np.array(preds)

# -------- Metrics --------
def compute_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        true: Ground truth values
        pred: Predicted values
        
    Returns:
        Dict[str, float]: Dictionary containing:
            - 'MAE': Mean Absolute Error
            - 'MSE': Mean Squared Error
            - 'RMSE': Root Mean Squared Error
    """
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}

# -------- Plotting --------
def plot_series(
    history: np.ndarray,
    forecast: np.ndarray,
    truth: Optional[np.ndarray],
    out_path: Path
):
    """
    Create a plot showing historical data, forecast, and ground truth.
    
    Args:
        history: Historical time series data
        forecast: Predicted values
        truth: Ground truth values (optional)
        out_path: Path to save the plot
    """
    n_hist = history.shape[0]
    xs = np.arange(n_hist)
    xf = np.arange(n_hist, n_hist + forecast.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(xs, history, label='History')
    plt.plot(xf, forecast, '--', label='Forecast')
    if truth is not None:
        plt.plot(xf, truth, ':', label='Truth')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    logging.info(f"Saved plot to {out_path}")
    plt.close()

# -------- Main Evaluation Pipeline --------

def evaluate(config: EvalConfig):
    """
    Run the complete evaluation pipeline.
    
    This function:
    1. Sets up logging and device
    2. Loads model checkpoint and scaler
    3. Instantiates the model
    4. Loads and normalizes historical data
    5. Generates forecasts
    6. Computes metrics if ground truth is available
    7. Creates visualization plots
    8. Exports the model to TorchScript format
    
    Args:
        config: Evaluation configuration object
    """
    logger = setup_logger(config.log_level)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ensure_dir(config.results_dir)

    # Load checkpoint & scaler
    ckpt, device = load_checkpoint(config.checkpoint, device)
    scaler_path = config.scaler or config.checkpoint.with_name(
        config.checkpoint.stem.replace('_best', '') + '_scaler.joblib'
    )
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded scaler from {scaler_path}")

    # Instantiate model
    model, lookback = instantiate_model(ckpt, device)

    # Load and normalize history
    hist = load_series(config.hist_data)
    hist_norm = scaler.transform(hist.reshape(-1, 1)).ravel()

    # Forecast
    preds_norm = recursive_forecast(model, hist_norm, lookback, config.n_steps, device)
    preds = scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel()

    # Evaluation metrics
    metrics = {}
    truth = None
    if config.test_data:
        test = load_series(config.test_data)
        truth = test[:config.n_steps]
        metrics = compute_metrics(truth, preds)
        metrics_path = config.results_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}: {metrics}")

    # Plot
    plot_path = config.results_dir / 'forecast.png'
    plot_series(hist, preds, truth, plot_path)

    # Export TorchScript
    ts_path = config.results_dir / (config.checkpoint.stem + '_ts.pt')
    example = torch.randn(1, lookback, 1).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(ts_path)
    logger.info(f"Exported TorchScript model to {ts_path}")

# -------- CLI --------

def parse_args() -> EvalConfig:
    """
    Parse command line arguments for evaluation.
    
    Returns:
        EvalConfig: Configuration object with parsed arguments
    """
    p = argparse.ArgumentParser(description="Evaluate GRU forecasting model")
    p.add_argument('--checkpoint', type=Path, required=True, help='Model checkpoint path')
    p.add_argument('--scaler', type=Path, default=None, help='Scaler joblib file')
    p.add_argument('--hist-data', type=Path, default='data/Xtrain.mat', help='Historical data file')
    p.add_argument('--test-data', type=Path, default=None, help='Test data file')
    p.add_argument('--n-steps', type=int, default=100, help='Forecast horizon')
    p.add_argument('--results-dir', type=Path, default=Path('eval_results'), help='Output directory')
    p.add_argument('--log-level', type=int, default=logging.INFO, help='Logging level')
    args = p.parse_args()
    return EvalConfig(**vars(args))

if __name__ == '__main__':
    cfg = parse_args()
    evaluate(cfg)
