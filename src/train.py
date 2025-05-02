"""
This module implements the training pipeline for the GRU forecasting model.
It includes functionality for hyperparameter optimization using Optuna,
cross-validation, and final model training with early stopping.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict

import joblib
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import optuna

from data_loader import load_series, scale_series, TimeSeriesDataset
from models import GRUForecast

# -------- Configuration --------

@dataclass
class Config:
    """
    Configuration class for model training.
    
    This dataclass holds all the necessary parameters for the training pipeline:
    - Data paths and output directories
    - Cross-validation settings
    - Training parameters
    - Early stopping criteria
    """
    data_path: str = 'data/Xtrain.mat'  # Path to training data
    output_dir: str = 'models'  # Directory to save models and results
    base_name: str = 'gru_tuned'  # Base name for saved files
    cv_trials: int = 50  # Number of hyperparameter optimization trials
    cv_folds: int = 5  # Number of cross-validation folds
    cv_epochs: int = 50  # Maximum epochs per CV fold
    cv_patience: int = 5  # Early stopping patience for CV
    final_epochs: int = 150  # Maximum epochs for final training
    final_patience: int = 15  # Early stopping patience for final training
    val_split: float = 0.2  # Validation set size ratio

# -------- Logger & Device --------

def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with the specified level.
    
    Args:
        level: Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    return logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Get the appropriate device (CPU or GPU) for training.
    
    Returns:
        torch.device: CUDA device if available, else CPU
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = setup_logger()

# -------- Checkpoint & Artifact Utilities --------

def ensure_dir(path: Path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def load_model_if_exists(
    ckpt_path: Path,
    hyperparams: Dict,
    device: torch.device
) -> torch.nn.Module:
    """
    Load a model from checkpoint if it exists and matches the hyperparameters.
    
    Args:
        ckpt_path: Path to the checkpoint file
        hyperparams: Dictionary of model hyperparameters
        device: Device to load the model onto
        
    Returns:
        torch.nn.Module: Loaded model or None if checkpoint doesn't exist/mismatch
    """
    if not ckpt_path.exists():
        return None
    checkpoint = torch.load(ckpt_path, map_location=device)
    if checkpoint.get('hyperparams') != hyperparams:
        return None
    model = GRUForecast(
        hidden_size=hyperparams['hidden_size'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams.get('dropout', 0.0)
    ).to(device)
    model.load_state_dict(checkpoint['model_state'])
    logger.info(f"Loaded existing model from {ckpt_path}")
    return model


def save_checkpoint(
    ckpt_path: Path,
    model: nn.Module,
    hyperparams: Dict,
    val_loss: float
):
    """
    Save a model checkpoint with its hyperparameters and validation loss.
    
    Args:
        ckpt_path: Path to save the checkpoint
        model: Model to save
        hyperparams: Dictionary of model hyperparameters
        val_loss: Validation loss achieved by the model
    """
    torch.save({
        'model_state': model.state_dict(),
        'hyperparams': hyperparams,
        'val_loss': val_loss
    }, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

# -------- Training & Evaluation --------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        loader: DataLoader providing training batches
        criterion: Loss function
        optimizer: Optimizer for updating model parameters
        device: Device to run training on
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: DataLoader providing evaluation batches
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        float: Average evaluation loss
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            losses.append(criterion(model(X), y).item())
    return float(np.mean(losses))

# -------- Hyperparameter Optimization --------

def objective(
    trial: optuna.trial.Trial,
    data: np.ndarray,
    config: Config
) -> float:
    """
    Objective function for hyperparameter optimization.
    
    This function:
    1. Suggests hyperparameters using Optuna
    2. Performs k-fold cross-validation
    3. Trains and evaluates the model for each fold
    4. Returns the mean validation loss across folds
    
    Args:
        trial: Optuna trial object
        data: Training data
        config: Training configuration
        
    Returns:
        float: Mean validation loss across folds
    """
    # Suggest hyperparameters
    params = {
        'lookback': trial.suggest_int('lookback', 100, 200, step=10),
        'batch_size': trial.suggest_categorical('batch_size', [4,8,16,32,64,128]),
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=32),
        'num_layers': trial.suggest_int('num_layers', 1, 6),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6, step=0.1),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    }
    device = get_device()
    ds = TimeSeriesDataset(data, params['lookback'])
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)

    fold_vals = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(ds), start=1):
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=params['batch_size'])

        model = GRUForecast(
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout']
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=max(1, config.cv_patience//2),
            factor=0.5
        )

        best_val, no_improve = float('inf'), 0
        for epoch in range(1, config.cv_epochs + 1):
            train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
            else:
                no_improve += 1
                if no_improve >= config.cv_patience:
                    break

        fold_vals.append(best_val)
        trial.report(best_val, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_vals))


def optimize_hyperparameters(
    data: np.ndarray,
    config: Config
) -> Tuple[Dict, optuna.Study]:
    """
    Run hyperparameter optimization using Optuna.
    
    This function:
    1. Creates an Optuna study with TPE sampler and Hyperband pruner
    2. Runs the optimization for the specified number of trials
    3. Saves the study and best parameters
    
    Args:
        data: Training data
        config: Training configuration
        
    Returns:
        Tuple[Dict, optuna.Study]:
            - Best hyperparameters found
            - Complete Optuna study object
    """
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.HyperbandPruner()
    )
    study.optimize(lambda t: objective(t, data, config), n_trials=config.cv_trials)

    # Save study and best parameters
    out_dir = Path(config.output_dir)
    ensure_dir(out_dir)
    study_path = out_dir / f"{config.base_name}_study.pkl"
    joblib.dump(study, study_path)
    logger.info(f"Optuna study saved to {study_path}")

    params_path = out_dir / f"{config.base_name}_best_params.json"
    with open(params_path, 'w') as fp:
        json.dump(study.best_trial.params, fp, indent=2)
    logger.info(f"Best params saved to {params_path}")

    return study.best_trial.params, study

# -------- Final Training --------

def train_final(
    data: np.ndarray,
    scaler,
    best_params: dict,
    config: Config
) -> torch.nn.Module:
    """
    Train the final model using the best hyperparameters.
    
    This function:
    1. Ensures output directory exists
    2. Loads or saves the data scaler
    3. Loads an existing model checkpoint if available
    4. Creates train/validation splits
    5. Trains the model with early stopping
    6. Saves checkpoints and final model
    """
    # Select device and prepare output directory
    device = get_device()
    out_dir = Path(config.output_dir)
    ensure_dir(out_dir)

    # Define paths for checkpoint and scaler
    ckpt_path = out_dir / f"{config.base_name}_best.pth"
    scaler_path = out_dir / f"{config.base_name}_scaler.joblib"

    # -------------------------------------------------------------------------
    # 1) Ensure scaler on disk: load existing or save new one
    # -------------------------------------------------------------------------
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded existing scaler from {scaler_path}")
    else:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved new scaler to {scaler_path}")

    # -------------------------------------------------------------------------
    # 2) Load pretrained model if checkpoint matches hyperparameters
    # -------------------------------------------------------------------------
    model = load_model_if_exists(ckpt_path, best_params, device)
    if model is not None:
        logger.info("Using pretrained model, skipping final training.")
        return model

    # -------------------------------------------------------------------------
    # 3) Prepare dataset and data loaders
    # -------------------------------------------------------------------------
    ds = TimeSeriesDataset(data, best_params['lookback'])
    val_size = int(len(ds) * config.val_split)
    train_ds, val_ds = random_split(
        ds,
        [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=best_params['batch_size'],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=best_params['batch_size']
    )

    # -------------------------------------------------------------------------
    # 4) Initialize model, loss, optimizer, and learning rate scheduler
    # -------------------------------------------------------------------------
    model = GRUForecast(
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params.get('dropout', 0.0)
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = getattr(optim, best_params['optimizer'])(
        model.parameters(), lr=best_params['lr']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=max(1, config.final_patience // 2),
        factor=0.5
    )

    # -------------------------------------------------------------------------
    # 5) Training loop with early stopping
    # -------------------------------------------------------------------------
    best_val = float('inf')
    no_improve = 0
    for epoch in range(1, config.final_epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        # Evaluate on validation set
        val_loss = evaluate(
            model, val_loader, criterion, device
        )
        # Adjust learning rate on plateau
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )

        # Check for improvement and save checkpoint
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(ckpt_path, model, best_params, val_loss)
        else:
            no_improve += 1
            if no_improve >= config.final_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs (patience={config.final_patience})."
                )
                break

    # Return the trained model
    return model

# -------- Main Entry Point Point --------

def main():
    config = Config()
    logger.info("Loading and preprocessing data...")
    series = load_series(config.data_path)
    norm, scaler = scale_series(series)

    study_path = Path(config.output_dir) / f"{config.base_name}_study.pkl"
    params_path = Path(config.output_dir) / f"{config.base_name}_best_params.json"
    if study_path.exists() and params_path.exists():
        logger.info("Loading existing study and best parameters...")
        study = joblib.load(study_path)
        with open(params_path, 'r') as fp:
            best_params = json.load(fp)
    else:
        logger.info("Starting hyperparameter optimization...")
        best_params, study = optimize_hyperparameters(norm, config)

    logger.info("Training final model...")
    model = train_final(norm, scaler, best_params, config)
    logger.info("Training complete!")

if __name__ == '__main__':
    main()
