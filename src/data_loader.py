"""
This module provides utilities for loading, preprocessing, and preparing time series data
for training and evaluation. It includes functions for data loading, normalization,
and creating sliding window datasets suitable for time series forecasting.
"""

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, TensorDataset
from typing import Tuple, Optional


def load_series(path: str = "data/Xtrain.mat") -> np.ndarray:
    """
    Load the 1D series from a .mat file and return as a float32 NumPy array.
    
    This function handles loading MATLAB .mat files containing time series data.
    It supports multiple possible data keys in the .mat file and converts the
    data to a 1D float32 array for efficient processing.
    
    Args:
        path: Path to the .mat file containing the time series data
        
    Returns:
        np.ndarray: 1D array of time series data with dtype float32
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        KeyError: If the expected data key is not found in the .mat file
    """
    try:
        mat = loadmat(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")
    
    # Try different possible keys for the data
    arr = mat.get('Xtrain', mat.get('data', None))
    if arr is None:
        raise KeyError(f"Could not find 'Xtrain' or 'data' in {path}")
    
    return arr.ravel().astype(np.float32)


def scale_series(series: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Fit a MinMaxScaler to [0,1] on the series and return (normalized, scaler).
    
    This function normalizes the time series data to the range [0,1] using
    scikit-learn's MinMaxScaler. The scaler is fitted to the data and returned
    for later use in inverse transformation.
    
    Args:
        series: 1D array of time series data
        
    Returns:
        Tuple[np.ndarray, MinMaxScaler]: 
            - Normalized data array in range [0,1]
            - Fitted scaler object for inverse transformation
        
    Raises:
        ValueError: If input is not a 1D array
    """
    if not isinstance(series, np.ndarray):
        series = np.asarray(series)
    
    if series.ndim != 1:
        raise ValueError(f"Expected 1D array, got {series.ndim}D")
    
    series = series.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    norm = scaler.fit_transform(series).ravel()
    return norm, scaler


def create_windows(series: np.ndarray, lookback: int) -> TensorDataset:
    """
    Convert a 1D array into sliding-window samples efficiently using numpy.
    
    This function creates a dataset of sliding windows from the time series,
    where each window contains 'lookback' previous time steps and the target
    is the next value in the sequence. The implementation uses numpy's
    sliding_window_view for efficient memory usage.
    
    Args:
        series: 1D array of time series data
        lookback: Number of time steps to look back for each sample
        
    Returns:
        TensorDataset: Dataset containing:
            - X: Input windows of shape [N, lookback, 1]
            - y: Target values of shape [N, 1]
            where N is the number of samples
        
    Raises:
        ValueError: If series length is less than lookback
    """
    series = np.asarray(series, dtype=np.float32)
    N = len(series) - lookback
    
    if N <= 0:
        raise ValueError(f"Series length ({len(series)}) must exceed lookback ({lookback})")
    
    # Create sliding window view (NumPy>=1.20)
    windows = np.lib.stride_tricks.sliding_window_view(series, window_shape=lookback)[:N]
    # Ensure writable contiguous array to avoid PyTorch warning
    windows = np.ascontiguousarray(windows)
    targets = series[lookback:lookback+N]

    X = torch.from_numpy(windows).unsqueeze(-1)  # shape (N, lookback, 1)
    y = torch.from_numpy(targets).unsqueeze(-1)  # shape (N, 1)
    return TensorDataset(X, y)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapping a precomputed TensorDataset of sliding windows.
    
    This class provides a PyTorch Dataset interface for the sliding window
    time series data. It wraps the TensorDataset created by create_windows
    and provides standard PyTorch Dataset functionality.
    
    Args:
        series: 1D array of time series data
        lookback: Number of time steps to look back for each sample
    """
    def __init__(self, series: np.ndarray, lookback: int):
        """
        Initialize the dataset with time series data and lookback window size.
        
        Args:
            series: 1D array of time series data
            lookback: Number of time steps to look back for each sample
        """
        self.dataset = create_windows(series, lookback)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Input window of shape [lookback, 1]
                - Target value of shape [1]
        """
        return self.dataset[idx]
