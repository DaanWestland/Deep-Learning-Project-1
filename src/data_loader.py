"""
This module provides utilities for loading, preprocessing, and preparing time series data
for training and evaluation. It includes functions for data loading, normalization,
and creating sliding window datasets suitable for both single-step and multi-step
(time-horizon or chunked) forecasting.
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Tuple


def load_series(path: str = "data/Xtrain.mat") -> np.ndarray:
    """
    Load the 1D series from a .mat file and return as a float32 NumPy array.

    Args:
        path: Path to the .mat file containing the time series data
    Returns:
        np.ndarray: 1D array of time series data with dtype float32
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If expected data key not found
    """
    try:
        mat = loadmat(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")

    arr = mat.get('Xtrain', mat.get('data', None))
    if arr is None:
        raise KeyError(f"Could not find 'Xtrain' or 'data' in {path}")

    return arr.ravel().astype(np.float32)


def scale_series(series: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize the series to [0,1] and return the normalized array and fitted scaler.

    Args:
        series: 1D array of time series data
    Returns:
        Tuple[np.ndarray, MinMaxScaler]
    Raises:
        ValueError: If input is not 1D
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim != 1:
        raise ValueError(f"Expected 1D array, got {series.ndim}D")

    series = series.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm = scaler.fit_transform(series).ravel()
    return norm, scaler


def create_windows(
    series: np.ndarray,
    lookback: int,
    horizon: int = 1
) -> TensorDataset:
    """
    Create sliding-window input-target pairs for single-step or multi-step forecasting.

    Args:
        series: 1D array of time series data
        lookback: Number of time steps to use as input
        horizon: Number of future steps to predict (multi-step). horizon=1 for one-step.
    Returns:
        TensorDataset containing:
            X: [N, lookback, 1]
            y: [N, horizon]  # multi-output
    Raises:
        ValueError: If series length is too short
    """
    series = np.asarray(series, dtype=np.float32)
    L = lookback
    H = horizon
    N = len(series) - L - H + 1
    if N <= 0:
        raise ValueError(
            f"Series length ({len(series)}) must exceed lookback+ horizon-1 ({L + H - 1})"
        )

    # Input windows: shape (N, L)
    X = np.lib.stride_tricks.sliding_window_view(series, window_shape=L)[:N]
    X = np.ascontiguousarray(X)

    # Target windows: shape (N, H)
    # sliding on the future portion
    Y = np.lib.stride_tricks.sliding_window_view(series[L:], window_shape=H)[:N]
    Y = np.ascontiguousarray(Y)

    # Convert to tensors
    X_t = torch.from_numpy(X).unsqueeze(-1)       # (N, L, 1)
    y_t = torch.from_numpy(Y)                     # (N, H)
    return TensorDataset(X_t, y_t)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for sliding-window time series forecasting,
    supporting multi-step horizons (horizon>=1).

    Args:
        series: 1D array of time series data
        lookback: number of input time steps
        horizon: number of future steps to predict
    """
    def __init__(self, series: np.ndarray, lookback: int, horizon: int = 1):
        self.dataset = create_windows(series, lookback, horizon)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = self.dataset[idx]
        return X, y
