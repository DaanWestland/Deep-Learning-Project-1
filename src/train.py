"""
Training pipeline for time series forecasting models.
Includes hyperparameter optimization with Optuna and final model training.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union, List

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.cuda.amp import autocast as cuda_autocast, GradScaler as CudaGradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import TimeSeriesSplit
import optuna

from data_loader import TimeSeriesDataset, FeatureEngineer
from models import GRUForecast, GRUSeq2Seq


@dataclass
class Config:
    """Configuration for the training pipeline."""
    # Data paths
    data_path: str = 'data/Xtrain.mat'
    output_dir: str = 'models'
    base_name: str = 'gru'

    # Hyperparameter search
    cv_trials: int = 500
    cv_folds: int = 10
    cv_gap: int = 0
    cv_epochs: int = 100
    cv_patience: int = 20

    # Final training
    final_epochs: int = 500
    final_patience: int = 100
    val_split: float = 0.1

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False

    # Training
    grad_clip: Optional[float] = 1.0
    use_amp: bool = True


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


def build_opt_sched(
    model: nn.Module,
    params: Dict,
    steps_per_epoch: int,
    total_epochs: int
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """
    Build optimizer and learning rate scheduler based on parameters.
    
    Args:
        model: PyTorch model
        params: Dictionary of training parameters
        steps_per_epoch: Number of steps per epoch
        total_epochs: Total number of epochs
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Build optimizer
    opt_cls = getattr(optim, params.get('optimizer', 'AdamW'))
    optimizer = opt_cls(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params.get('weight_decay', 0.0)
    )

    # Build scheduler if specified
    sched_name = params.get('scheduler')
    scheduler = None

    if sched_name == 'OneCycleLR' and steps_per_epoch > 0:
        max_lr = max(params['lr'], params.get('onecycle_max_lr', params['lr']))
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_epochs * steps_per_epoch,
            pct_start=params.get('pct_start', 0.3),
            anneal_strategy='cos',
            div_factor=params.get('onecycle_div_factor', 25.0),
            final_div_factor=params.get('onecycle_final_div_factor', 1e4)
        )
    elif sched_name == 'CosineAnnealingWarmRestarts' and steps_per_epoch > 0:
        T_0 = max(1, params.get('T_0', 10) * steps_per_epoch)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=params.get('cosine_T_mult', 1),
            eta_min=params.get('cosine_eta_min_factor', 0.01) * params['lr']
        )
    elif sched_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params.get('plateau_factor', 0.1),
            patience=params.get('plateau_patience', 10),
            min_lr=params.get('plateau_min_lr', 1e-7)
        )

    if scheduler is None and sched_name not in [None, 'None']:
        logger.warning(f"Scheduler '{sched_name}' not created (steps_per_epoch={steps_per_epoch})")
    elif sched_name is None or sched_name == 'None':
        logger.info("No learning rate scheduler selected")

    return optimizer, scheduler


def objective(
    trial: optuna.Trial,
    series_source: Union[str, np.ndarray, pd.DataFrame, pd.Series],
    config: Config,
    fixed_model: str,
    device: torch.device
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        series_source: Source of time series data
        config: Training configuration
        fixed_model: Model type to use
        device: Device to train on
        
    Returns:
        Average validation loss across folds
    """
    # Suggest hyperparameters
    params = {
        'scale_method': trial.suggest_categorical('scale_method', ['minmax', 'standard', 'power', 'quantile', None]),
        'enforce_stat': trial.suggest_categorical('enforce_stat', [True, False]),
        'diff_order': trial.suggest_int('diff_order', 1, 2) if trial.suggest_categorical('enforce_stat', [True, False]) else 0,
        'use_feature_engineer': trial.suggest_categorical('use_feature_engineer', [True, False])
    }

    # Feature engineering parameters
    if params['use_feature_engineer']:
        num_lags = trial.suggest_int('num_lags', 0, 10)
        params['lags'] = list(range(1, num_lags + 1)) if num_lags > 0 else []
        
        num_roll_windows = trial.suggest_int('num_roll_windows_choices', 0, 2)
        possible_windows = [3, 7, 14, 21, 30]
        if num_roll_windows > 0:
            chosen_indices = np.random.choice(len(possible_windows), num_roll_windows, replace=False)
            params['rolling_windows'] = sorted([possible_windows[i] for i in chosen_indices])
        else:
            params['rolling_windows'] = []
    else:
        params['lags'], params['rolling_windows'] = [], []

    # Model-specific parameters
    if fixed_model == 'GRUForecast':
        params.update({
            'lr': trial.suggest_float('lr', 1e-4, 3e-2, log=True),
            'num_layers': trial.suggest_int('num_layers', 1, 6),
            'hidden_size': trial.suggest_int('hidden_size', 16, 128, step=16)
        })
    else:  # GRUSeq2Seq
        params.update({
            'lr': trial.suggest_float('lr', 5e-5, 1e-2, log=True),
            'num_layers': trial.suggest_int('num_layers', 1, 6),
            'hidden_size': trial.suggest_int('hidden_size', 16, 256, step=16)
        })

    # Common parameters
    params.update({
        'lookback': trial.suggest_int('lookback', 20, 150, step=10),
        'horizon': max(1, trial.suggest_int('horizon', 10, 50, step=10)),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'RMSprop']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True),
        'scheduler': trial.suggest_categorical('scheduler', ['ReduceLROnPlateau', 'OneCycleLR', 'CosineAnnealingWarmRestarts', None]),
        'tf_init': trial.suggest_float('tf_init', 0.2, 0.8, step=0.1) if fixed_model == 'GRUSeq2Seq' else 0.0
    })

    # Scheduler-specific parameters
    if params['scheduler'] == 'OneCycleLR':
        params.update({
            'pct_start': trial.suggest_float('pct_start', 0.1, 0.4, step=0.05),
            'onecycle_max_lr': trial.suggest_float('onecycle_max_lr', params['lr'] * 1.1, params['lr'] * 20, log=True),
            'onecycle_div_factor': trial.suggest_float('onecycle_div_factor', 5, 50, log=True)
        })
    elif params['scheduler'] == 'CosineAnnealingWarmRestarts':
        params.update({
            'T_0': trial.suggest_int('T_0', 5, max(6, config.cv_epochs // 3)),
            'cosine_T_mult': trial.suggest_categorical('cosine_T_mult', [1, 2]),
            'cosine_eta_min_factor': trial.suggest_float('cosine_eta_min_factor', 1e-3, 0.5, log=True)
        })
    elif params['scheduler'] == 'ReduceLROnPlateau':
        params.update({
            'plateau_factor': trial.suggest_float('plateau_factor', 0.1, 0.5, step=0.05),
            'plateau_patience': trial.suggest_int('plateau_patience', 3, max(4, config.cv_patience // 2))
        })

    params['model_type'] = fixed_model
    logger.info(f"[Trial {trial.number}] {fixed_model} params:\n{json.dumps(params, indent=2, sort_keys=True)}")

    # Create dataset
    engineer = FeatureEngineer(lags=params['lags'], rolling_windows=params['rolling_windows']) if params['use_feature_engineer'] else None
    
    try:
        ds = TimeSeriesDataset(
            series_source,
            lookback=params['lookback'],
            horizon=params['horizon'],
            enforce_stat=params['enforce_stat'],
            diff_order=params['diff_order'],
            scale_method=params['scale_method'],
            engineer=engineer
        )
        if len(ds) == 0:
            raise optuna.TrialPruned("Dataset empty")
        X_all, y_all = ds.X, ds.y
    except Exception as e:
        logger.error(f"Trial {trial.number}: Dataset creation error: {e}", exc_info=True)
        raise optuna.TrialPruned(f"Dataset error: {e}")

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=config.cv_folds, gap=config.cv_gap)
    grad_scaler = CudaGradScaler(enabled=(config.use_amp and device.type == 'cuda'))
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all), start=1):
        if len(train_idx) == 0 or len(val_idx) == 0:
            logger.warning(f"Trial {trial.number} Fold {fold}: Empty train/val set")
            continue

        # Prepare data
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_va, y_va = X_all[val_idx], y_all[val_idx]

        # Adjust batch size if needed
        batch_size = min(params['batch_size'], len(X_tr))
        drop_last = len(X_tr) > batch_size and len(X_tr) % batch_size != 0

        # Create data loaders
        tr_loader = DataLoader(
            torch.utils.data.TensorDataset(X_tr, y_tr),
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor
        )
        val_batch_size = min(batch_size * 2, len(X_va)) if len(X_va) > 0 else 1
        va_loader = DataLoader(
            torch.utils.data.TensorDataset(X_va, y_va),
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            prefetch_factor=config.prefetch_factor
        )

        if len(tr_loader) == 0 or len(va_loader) == 0:
            logger.warning(f"Trial {trial.number} Fold {fold}: Empty data loaders")
            continue

        # Create model
        input_features = X_all.size(2)
        if fixed_model == 'GRUForecast':
            model = GRUForecast(
                input_size=input_features,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                horizon=params['horizon']
            )
        else:  # GRUSeq2Seq
            model = GRUSeq2Seq(
                encoder_input_size=input_features,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                horizon=params['horizon'],
                decoder_input_size=1
            )
        model.to(device)

        # Setup training
        optimizer, scheduler = build_opt_sched(
            model,
            params,
            len(tr_loader),
            config.cv_epochs
        )
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        fold_loss = float('inf')

        for epoch in range(config.cv_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in tr_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                with cuda_autocast(enabled=config.use_amp):
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)

                if config.use_amp:
                    grad_scaler.scale(loss).backward()
                    if config.grad_clip is not None:
                        grad_scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), config.grad_clip)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    if config.grad_clip is not None:
                        clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                train_loss += loss.item()

                if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            train_loss /= len(tr_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in va_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
            val_loss /= len(va_loader)

            # Learning rate scheduling
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                fold_loss = val_loss
            else:
                patience_counter += 1
                if patience_counter >= config.cv_patience:
                    logger.info(f"Trial {trial.number} Fold {fold}: Early stopping at epoch {epoch + 1}")
                    break

        fold_losses.append(fold_loss)

    if not fold_losses:
        raise optuna.TrialPruned("No valid folds completed")

    return np.mean(fold_losses)


def optimize_hyperparameters(
    series_source: Union[str, np.ndarray, pd.DataFrame, pd.Series],
    config: Config,
    model_type: str,
    device: torch.device
) -> Dict:
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        series_source: Source of time series data
        config: Training configuration
        model_type: Type of model to optimize
        device: Device to train on
        
    Returns:
        Dictionary of best hyperparameters
    """
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, series_source, config, model_type, device),
        n_trials=config.cv_trials,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_params['model_type'] = model_type
    logger.info(f"Best parameters:\n{json.dumps(best_params, indent=2)}")

    return best_params


def train_final(
    series_source: Union[str, np.ndarray, pd.DataFrame, pd.Series],
    params: Dict,
    config: Config,
    model_type: str,
    device: torch.device,
    suffix: str = '_final'
) -> Optional[nn.Module]:
    """
    Train final model with best hyperparameters.
    
    Args:
        series_source: Source of time series data
        params: Best hyperparameters
        config: Training configuration
        model_type: Type of model to train
        device: Device to train on
        suffix: Suffix for output files
        
    Returns:
        Trained model
    """
    # Create dataset
    engineer = FeatureEngineer(lags=params['lags'], rolling_windows=params['rolling_windows']) if params['use_feature_engineer'] else None
    
    try:
        ds = TimeSeriesDataset(
            series_source,
            lookback=params['lookback'],
            horizon=params['horizon'],
            enforce_stat=params['enforce_stat'],
            diff_order=params['diff_order'],
            scale_method=params['scale_method'],
            engineer=engineer
        )
        if len(ds) == 0:
            raise ValueError("Dataset empty")
    except Exception as e:
        logger.error(f"Final training dataset creation error: {e}", exc_info=True)
        return None

    # Split data
    val_size = int(len(ds) * config.val_split)
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params['batch_size'] * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor
    )

    # Create model
    input_features = ds.X.size(2)
    if model_type == 'GRUForecast':
        model = GRUForecast(
            input_size=input_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            horizon=params['horizon']
        )
    else:  # GRUSeq2Seq
        model = GRUSeq2Seq(
            encoder_input_size=input_features,
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            horizon=params['horizon'],
            decoder_input_size=1
        )
    model.to(device)

    # Setup training
    optimizer, scheduler = build_opt_sched(
        model,
        params,
        len(train_loader),
        config.final_epochs
    )
    criterion = nn.MSELoss()
    grad_scaler = CudaGradScaler(enabled=(config.use_amp and device.type == 'cuda'))

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.final_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            with cuda_autocast(enabled=config.use_amp):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

            if config.use_amp:
                grad_scaler.scale(loss).backward()
                if config.grad_clip is not None:
                    grad_scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), config.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if config.grad_clip is not None:
                    clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            train_loss += loss.item()

            if isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
                scheduler.step()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += criterion(y_pred, y_batch).item()
        val_loss /= len(val_loader)

        # Learning rate scheduling
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= config.final_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}/{config.final_epochs} - "
                       f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if best_model_state is None:
        logger.error("No valid model state saved")
        return None

    # Save best model
    model.load_state_dict(best_model_state)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / f"{config.base_name}_tuned_best{suffix}.pth"
    torch.save({
        'model_state_dict': best_model_state,
        'model_params': params,
        'feature_names': ds.feature_names if hasattr(ds, 'feature_names') else None
    }, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save scaler
    if ds.scaler is not None:
        scaler_path = output_dir / f"{config.base_name}_tuned_scaler{suffix}.joblib"
        joblib.dump(ds.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

    # Save parameters
    params_path = output_dir / f"{config.base_name}_tuned_best_params{suffix}.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved parameters to {params_path}")

    return model


def main():
    """Main training pipeline."""
    # Setup
    config = Config()
    device = get_device()

    # Load data
    try:
        series_source = config.data_path
        logger.info(f"Loading data from {series_source}")
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    # Optimize hyperparameters
    try:
        best_params = optimize_hyperparameters(
            series_source,
            config,
            'GRUForecast',  # or 'GRUSeq2Seq'
            device
        )
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}", exc_info=True)
        return

    # Train final model
    try:
        model = train_final(
            series_source,
            best_params,
            config,
            'GRUForecast',  # or 'GRUSeq2Seq'
            device
        )
        if model is None:
            logger.error("Final training failed")
            return
    except Exception as e:
        logger.error(f"Final training failed: {e}", exc_info=True)
        return

    logger.info("Training pipeline completed successfully")


if __name__ == '__main__':
    main()
