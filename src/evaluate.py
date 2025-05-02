"""
This module provides functionality for evaluating trained GRU forecasting models.
It supports both GRUForecast (MIMO) and GRUSeq2Seq model types, handles multi-step
forecasting via direct or chunked inference, computes metrics, and generates plots.
"""

import argparse
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import joblib
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_series, scale_series
from models import GRUForecast, GRUSeq2Seq

# -------- Configuration --------
@dataclass
class EvalConfig:
    checkpoint: Path
    scaler: Optional[Path]
    hist_data: Path
    test_data: Optional[Path]
    n_steps: int
    results_dir: Path
    log_level: int = logging.INFO

# -------- Logger --------
def setup_logger(level: int) -> logging.Logger:
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    return logging.getLogger(__name__)

# -------- Utilities --------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# -------- Checkpoint Loading --------
def load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    logging.info(f"Loading checkpoint from {path}")
    return torch.load(path, map_location=device)

# -------- Model Instantiation --------
def instantiate_model(ckpt: Dict, device: torch.device) -> Tuple[nn.Module, Dict]:
    params = ckpt.get('hyperparams') or ckpt.get('model_params') or {}
    if 'model_type' not in params:
        # default to GRUForecast if unspecified
        params['model_type'] = 'GRUForecast'
    model_type = params['model_type']
    horizon = params.get('horizon', 1)
    # Instantiate correct class
    if model_type == 'GRUForecast':
        model = GRUForecast(
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.0),
            horizon=horizon
        )
    elif model_type == 'GRUSeq2Seq':
        model = GRUSeq2Seq(
            input_size=1,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.0),
            horizon=horizon
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    logging.info(f"Instantiated {model_type} with horizon={horizon}")
    return model, params

# -------- Forecasting --------
def forecast(model: nn.Module,
             init_seq: np.ndarray,
             params: Dict,
             n_steps: int,
             device: torch.device) -> np.ndarray:
    """
    Multi-step forecasting supporting:
      - GRUForecast (direct MIMO or chunked)
      - GRUSeq2Seq (chunked iterative)
    """
    lookback = params.get('lookback')
    horizon = params.get('horizon', 1)
    model_type = params.get('model_type')
    seq = init_seq.tolist()
    preds = []
    idx = 0

    # Define a helper to get next chunk
    def predict_chunk(window):
        x = torch.tensor(window, dtype=torch.float32).view(1, lookback, 1).to(device)
        with torch.no_grad():
            if model_type == 'GRUForecast':
                out = model(x)           # shape (1, horizon)
                chunk = out.cpu().numpy().ravel().tolist()
            else:
                out = model(x, target=None, teacher_forcing_prob=0.0)
                chunk = out.cpu().numpy().ravel().tolist()
        return chunk

    # If MIMO can predict full horizon at once
    if model_type == 'GRUForecast' and horizon >= n_steps:
        chunk = predict_chunk(seq[-lookback:])
        preds = chunk[:n_steps]
    else:
        # Chunked iterative forecasting
        while idx < n_steps:
            chunk = predict_chunk(seq[-lookback:])
            step = min(horizon, n_steps - idx)
            preds.extend(chunk[:step])
            # update history
            seq.extend(chunk[:step])
            idx += step
    return np.array(preds)

# -------- Metrics & Plotting --------
def compute_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': float(np.sqrt(mse))}


def plot_series(history: np.ndarray,
                forecast: np.ndarray,
                truth: Optional[np.ndarray],
                out_path: Path):
    n_hist = len(history)
    xs = np.arange(n_hist)
    xf = np.arange(n_hist, n_hist + len(forecast))
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
    logging.info(f"Plot saved to {out_path}")
    plt.close()

# -------- Evaluation Pipeline --------
def evaluate(config: EvalConfig):
    logger = setup_logger(config.log_level)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(config.results_dir)

    # Load checkpoint & scaler
    ckpt = load_checkpoint(config.checkpoint, device)
    scaler_path = (config.scaler if config.scaler
                   else config.checkpoint.parent / f"{config.checkpoint.stem.replace('_best','')}_scaler.joblib")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded scaler from {scaler_path}")

    # Instantiate model
    model, params = instantiate_model(ckpt, device)

    # Load and scale history
    hist = load_series(config.hist_data)
    hist_norm, _ = scale_series(hist)

    # Forecast
    preds_norm = forecast(model, hist_norm, params, config.n_steps, device)
    preds = scaler.inverse_transform(preds_norm.reshape(-1, 1)).ravel()

    # Evaluate metrics if test_data provided
    metrics = {}
    truth = None
    if config.test_data:
        test = load_series(config.test_data)
        truth = test[:config.n_steps]
        metrics = compute_metrics(truth, preds)
        (config.results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        logger.info(f"Metrics: {metrics}")

    # Plot results
    plot_series(hist, preds, truth, config.results_dir / 'forecast.png')

    # TorchScript export
    ts_path = config.results_dir / f"{config.checkpoint.stem}_ts.pt"
    example = torch.randn(1, params['lookback'], 1).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(ts_path)
    logger.info(f"Saved TorchScript model to {ts_path}")

# -------- CLI --------
def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate GRU forecasting model")
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scaler', type=Path, default=None)
    p.add_argument('--hist-data', type=Path, default=Path('data/Xtrain.mat'))
    p.add_argument('--test-data', type=Path, default=None)
    p.add_argument('--n-steps', type=int, default=200)
    p.add_argument('--results-dir', type=Path, default=Path('eval_results'))
    p.add_argument('--log-level', type=int, default=logging.INFO)
    args = p.parse_args()
    return EvalConfig(**vars(args))

if __name__ == '__main__':
    cfg = parse_args()
    evaluate(cfg)
