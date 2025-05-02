"""
This module implements the training pipeline for GRU-based forecasting models.
It supports both single-step/multi-step (MIMO) and Seq2Seq GRU architectures.
Hyperparameter optimization is done via Optuna, with k-fold CV.
Final model is trained on full data using the best hyperparameters,
with early stopping, checkpointing, and progress logging.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import optuna

from data_loader import load_series, scale_series, TimeSeriesDataset
from models import GRUForecast, GRUSeq2Seq

# -------- Configuration --------
@dataclass
class Config:
    data_path: str = 'data/Xtrain.mat'
    output_dir: str = 'models'
    base_name: str = 'gru_tuned'
    cv_trials: int = 50
    cv_folds: int = 5
    cv_epochs: int = 50
    cv_patience: int = 5
    final_epochs: int = 150
    final_patience: int = 15
    val_split: float = 0.2

# -------- Logger & Device --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Return available device: prints running device (CPU vs CUDA),
    and GPU details if available.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    if device.type == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}, GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    return device

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

# -------- Objective for Optuna --------
def objective(trial: optuna.trial.Trial, data: np.ndarray, config: Config) -> float:
    # Suggest hyperparameters
    params = {
        'model_type': trial.suggest_categorical('model_type', ['GRUForecast', 'GRUSeq2Seq']),
        'lookback': trial.suggest_int('lookback', 100, 200, step=10),
        'horizon': trial.suggest_int('horizon', 1, 30, step=5),
        'batch_size': trial.suggest_categorical('batch_size', [4, 8, 16, 32, 48, 64]),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'hidden_size': trial.suggest_int('hidden_size', 16, 228, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 8, step=1),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW']),
        'tf_prob': trial.suggest_float('tf_prob', 0.0, 1.0)
    }
    logger.info(f"Trial {trial.number}: params={params}")
    device = get_device()

    # Prepare dataset and cross-validation
    ds = TimeSeriesDataset(data, params['lookback'], params['horizon'])
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
    fold_losses = []
    global_step = 0

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(ds))), start=1):
        logger.info(f" Trial {trial.number} - Fold {fold}/{config.cv_folds}")
        train_ds = Subset(ds, train_idx)
        val_ds   = Subset(ds, val_idx)
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=params['batch_size'])

        # Instantiate model for this fold
        if params['model_type'] == 'GRUForecast':
            model = GRUForecast(
                input_size=1,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                horizon=params['horizon']
            ).to(device)
        else:
            model = GRUSeq2Seq(
                input_size=1,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                horizon=params['horizon']
            ).to(device)

        criterion = nn.MSELoss()
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=max(1, config.cv_patience//2),
            factor=0.5
        )

        best_fold = float('inf')
        no_improve = 0
        # Epoch loop
        for epoch in range(1, config.cv_epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                if params['model_type'] == 'GRUSeq2Seq':
                    pred = model(X, target=y, teacher_forcing_prob=params['tf_prob'])
                else:
                    pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(device), y.to(device)
                    if params['model_type'] == 'GRUSeq2Seq':
                        pred = model(X, target=None, teacher_forcing_prob=0.0)
                    else:
                        pred = model(X)
                    val_loss += criterion(pred, y).item()
            val_loss /= len(val_loader)

            # Report to Optuna for pruning
            global_step += 1
            trial.report(val_loss, global_step)
            logger.info(f"  Epoch {epoch}/{config.cv_epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
            scheduler.step(val_loss)
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at fold {fold}, epoch {epoch}")
                raise optuna.TrialPruned()

            # Early stopping per fold
            if val_loss < best_fold:
                best_fold = val_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= config.cv_patience:
                    logger.info(f"Early stopping fold {fold} at epoch {epoch}")
                    break

        fold_losses.append(best_fold)
    # Compute mean of fold best losses
    mean_loss = float(np.mean(fold_losses))
    logger.info(f"Trial {trial.number} complete. Mean CV loss: {mean_loss:.6f}")
    return mean_loss

# -------- Hyperparameter Optimization --------
def optimize_hyperparameters(data: np.ndarray, config: Config) -> Tuple[Dict, optuna.Study]:
    ensure_dir(Path(config.output_dir))
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(lambda t: objective(t, data, config), n_trials=config.cv_trials)

    out_dir = Path(config.output_dir)
    # Save study
    study_path = out_dir / f"{config.base_name}_study.pkl"
    joblib.dump(study, study_path)
    logger.info(f"Optuna study saved to {study_path}")
    # Save best params
    params_path = out_dir / f"{config.base_name}_best_params.json"
    with open(params_path, 'w') as fp:
        json.dump(study.best_trial.params, fp, indent=2)
    logger.info(f"Best params saved to {params_path}")
    return study.best_trial.params, study

# -------- Final Training --------
def train_final(data: np.ndarray, scaler, best_params: Dict, config: Config) -> torch.nn.Module:
    device = get_device()
    out_dir = Path(config.output_dir)
    ensure_dir(out_dir)

    # Save or load scaler
    scaler_path = out_dir / f"{config.base_name}_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
    else:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

    # Define checkpoint path
    ckpt_name = f"{config.base_name}_{best_params['model_type']}_h{best_params['horizon']}_best.pth"
    ckpt_path = out_dir / ckpt_name

    # Load existing if matching
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt.get('hyperparams') == best_params:
            logger.info(f"Using existing model from {ckpt_path}")
            model = GRUForecast(input_size=1,
                                hidden_size=best_params['hidden_size'],
                                num_layers=best_params['num_layers'],
                                dropout=best_params['dropout'],
                                horizon=best_params['horizon']) if best_params['model_type']=='GRUForecast' else GRUSeq2Seq(
                                    input_size=1,
                                    hidden_size=best_params['hidden_size'],
                                    num_layers=best_params['num_layers'],
                                    dropout=best_params['dropout'],
                                    horizon=best_params['horizon']
                                )
            model.load_state_dict(ckpt['model_state'])
            return model.to(device)

    # Prepare dataset and loaders
    ds = TimeSeriesDataset(data, best_params['lookback'], best_params['horizon'])
    val_size = int(len(ds) * config.val_split)
    train_ds, val_ds = random_split(
        ds,
        [len(ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=best_params['batch_size'])

    # Instantiate model
    if best_params['model_type'] == 'GRUForecast':
        model = GRUForecast(
            input_size=1,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            horizon=best_params['horizon']
        ).to(device)
    else:
        model = GRUSeq2Seq(
            input_size=1,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout'],
            horizon=best_params['horizon']
        ).to(device)

    criterion = nn.MSELoss()
    optimizer = getattr(optim, best_params['optimizer'])(model.parameters(), lr=best_params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=max(1, config.final_patience//2),
        factor=0.5
    )

    # Final training loop
    logger.info("Starting final training...")
    best_val = float('inf')
    no_improve = 0
    for epoch in range(1, config.final_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            if best_params['model_type'] == 'GRUSeq2Seq':
                pred = model(X, target=y, teacher_forcing_prob=best_params['tf_prob'])
            else:
                pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                if best_params['model_type'] == 'GRUSeq2Seq':
                    pred = model(X, target=None, teacher_forcing_prob=0.0)
                else:
                    pred = model(X)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        logger.info(f"Epoch {epoch}/{config.final_epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

        # Checkpointing
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'hyperparams': best_params,
                'val_loss': val_loss
            }, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= config.final_patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={config.final_patience})")
                break

    logger.info("Final training complete.")
    return model

# -------- Main --------
def main():
    config = Config()
    logger.info("Loading data...")
    series = load_series(config.data_path)
    norm, scaler = scale_series(series)

    # Optuna study and parameters
    out_dir = Path(config.output_dir)
    study_file = out_dir / f"{config.base_name}_study.pkl"
    params_file = out_dir / f"{config.base_name}_best_params.json"
    if study_file.exists() and params_file.exists():
        logger.info("Loading existing study and parameters...")
        study = joblib.load(study_file)
        best_params = json.loads(params_file.read_text())
    else:
        logger.info("Starting hyperparameter optimization...")
        best_params, study = optimize_hyperparameters(norm, config)

    logger.info(f"Best hyperparameters: {best_params}")
    logger.info("Commencing final model training...")
    model = train_final(norm, scaler, best_params, config)
    logger.info("All done!")

if __name__ == '__main__':
    main()
