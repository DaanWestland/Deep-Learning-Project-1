"""
Evaluation pipeline for time series forecasting models.
Includes loading models, making predictions, and computing metrics.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from scipy.io import loadmat

from data_loader import TimeSeriesDataset, FeatureEngineer
from models import GRUForecast, GRUSeq2Seq
from metrics import compute_metrics


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Get the appropriate torch device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    return device


def load_raw_series(data_path: str) -> np.ndarray:
    """
    Load raw time series data from .mat file.
    
    Args:
        data_path: Path to .mat file
        
    Returns:
        Raw time series data as numpy array
        
    Raises:
        FileNotFoundError: If data file not found
        KeyError: If required key not found in .mat file
    """
    try:
        mat = loadmat(data_path)
        key = "Xtrain" if "Xtrain" in mat else "data" if "data" in mat else None
        if key is None:
            raise KeyError(f"MAT file {data_path} missing 'Xtrain' or 'data'")
        series = mat[key].ravel().astype(np.float32)
        logger.info(f"Loaded data from {data_path}, shape: {series.shape}")
        return series
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")


def load_checkpoint(
    model_path: str,
    device: torch.device
) -> Tuple[nn.Module, Dict, Optional[List[str]]]:
    """
    Load model checkpoint and parameters.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, parameters, feature_names)
        
    Raises:
        FileNotFoundError: If checkpoint not found
        KeyError: If required keys missing from checkpoint
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        params = checkpoint['model_params']
        feature_names = checkpoint.get('feature_names')
        
        # Create model
        if params['model_type'] == 'GRUForecast':
            model = GRUForecast(
                input_size=params['input_size'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                horizon=params['horizon']
            )
        else:  # GRUSeq2Seq
            model = GRUSeq2Seq(
                encoder_input_size=params['input_size'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                horizon=params['horizon'],
                decoder_input_size=1
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        return model, params, feature_names
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    except KeyError as e:
        raise KeyError(f"Missing key in checkpoint: {e}")
    except Exception as e:
        raise Exception(f"Error loading checkpoint: {e}")


def instantiate_model(
    params: Dict,
    input_size: int,
    device: torch.device
) -> nn.Module:
    """
    Create model instance from parameters.
    
    Args:
        params: Model parameters
        input_size: Input feature size
        device: Device to create model on
        
    Returns:
        Model instance
    """
    if params['model_type'] == 'GRUForecast':
        model = GRUForecast(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            horizon=params['horizon']
        )
    else:  # GRUSeq2Seq
        model = GRUSeq2Seq(
            encoder_input_size=input_size,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            horizon=params['horizon'],
            decoder_input_size=1
        )
    
    model.to(device)
    model.eval()
    return model


def preprocess_eval_data(
    series: np.ndarray,
    params: Dict,
    scaler_path: Optional[str] = None
) -> Tuple[TimeSeriesDataset, Optional[object]]:
    """
    Preprocess evaluation data.
    
    Args:
        series: Raw time series data
        params: Model parameters
        scaler_path: Path to saved scaler
        
    Returns:
        Tuple of (dataset, scaler)
    """
    # Load scaler if provided
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            logger.warning(f"Error loading scaler: {e}")
    
    # Create feature engineer if needed
    engineer = None
    if params.get('use_feature_engineer'):
        engineer = FeatureEngineer(
            lags=params.get('lags', []),
            rolling_windows=params.get('rolling_windows', [])
        )
    
    # Create dataset
    ds = TimeSeriesDataset(
        series,
        lookback=params['lookback'],
        horizon=params['horizon'],
        enforce_stat=params.get('enforce_stat', True),
        diff_order=params.get('diff_order', 1),
        scale_method=params.get('scale_method'),
        engineer=engineer,
        scaler=scaler
    )
    
    return ds, scaler


def forecast(
    model: nn.Module,
    ds: TimeSeriesDataset,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Generate forecasts using model.
    
    Args:
        model: Trained model
        ds: Preprocessed dataset
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Array of forecasts
    """
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    forecasts = []
    with torch.no_grad():
        for X_batch in loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            forecasts.append(y_pred.cpu().numpy())
    
    return np.concatenate(forecasts, axis=0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {}
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    metrics['MSE'] = mse
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    metrics['MAE'] = mae
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['MAPE'] = mape
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    metrics['R2'] = r2
    
    return metrics


def plot_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = 'Forecast vs Actual'
) -> None:
    """
    Plot true vs predicted values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='blue')
    plt.plot(y_pred, label='Forecast', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def evaluate_model(
    model_path: str,
    data_path: str,
    scaler_path: Optional[str] = None,
    output_dir: str = 'eval_results',
    plot: bool = True
) -> Dict[str, float]:
    """
    Evaluate model on test data.
    
    Args:
        model_path: Path to model checkpoint
        data_path: Path to test data
        scaler_path: Path to saved scaler
        output_dir: Directory to save results
        plot: Whether to generate plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Setup
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data and model
    series = load_raw_series(data_path)
    model, params, feature_names = load_checkpoint(model_path, device)
    
    # Preprocess data
    ds, scaler = preprocess_eval_data(series, params, scaler_path)
    
    # Generate forecasts
    forecasts = forecast(model, ds, device)
    
    # Compute metrics
    metrics = compute_metrics(ds.y.numpy(), forecasts)
    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    
    # Save results
    results_path = output_dir / f"{Path(model_path).stem}_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {results_path}")
    
    # Generate plots
    if plot:
        plot_path = output_dir / f"{Path(model_path).stem}_forecast.png"
        plot_series(
            ds.y.numpy(),
            forecasts,
            save_path=plot_path,
            title=f"Forecast vs Actual ({Path(model_path).stem})"
        )
    
    return metrics


def main():
    """Main evaluation pipeline."""
    # Configuration
    model_path = 'models/gru_tuned_best.pth'
    data_path = 'data/Xtest.mat'
    scaler_path = 'models/gru_tuned_scaler.joblib'
    output_dir = 'eval_results'
    
    try:
        metrics = evaluate_model(
            model_path=model_path,
            data_path=data_path,
            scaler_path=scaler_path,
            output_dir=output_dir,
            plot=True
        )
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)


if __name__ == '__main__':
    main()
