import numpy as np
import pandas as pd
import logging
from scipy.io import loadmat
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, PowerTransformer, QuantileTransformer
)
import torch
from torch.utils.data import Dataset
from typing import (
    Callable, List, Optional, Union, Tuple, Dict
)

# Setup logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s %(levelname)s data_loader] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def load_series(path: str) -> np.ndarray:
    """
    Load time series data from a .mat file.
    
    Args:
        path (str): Path to the .mat file containing the time series data.
        
    Returns:
        np.ndarray: 1D array of time series data with dtype float32.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        KeyError: If no suitable data key/array is found.
    """
    try:
        mat = loadmat(path)
    except FileNotFoundError:
        logger.error(f"Data file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Could not load .mat file {path}: {e}")
        raise

    arr = mat.get('Xtest', mat.get('Xtrain', mat.get('data', None)))
    
    if arr is None:
        logger.warning(f"Standard keys ('Xtest', 'Xtrain', 'data') not found in {path}. Attempting to find any suitable array.")
        available_keys = [k for k in mat if not k.startswith('__')]
        for key in available_keys:
            potential_arr = mat[key]
            if isinstance(potential_arr, np.ndarray) and potential_arr.ndim >= 1 and np.issubdtype(potential_arr.dtype, np.number):
                logger.info(f"Found numerical array under key: '{key}'. Using this data.")
                arr = potential_arr
                break
        if arr is None:
            raise KeyError(f"Could not find suitable numerical array in {path}. Available non-private keys: {available_keys}")

    return arr.ravel().astype(np.float32)

def adf_kpss_test(series: np.ndarray) -> Tuple[float, float]:
    """
    Perform ADF and KPSS tests for stationarity.
    
    Args:
        series (np.ndarray): Time series data to test.
        
    Returns:
        Tuple[float, float]: p-values for ADF and KPSS tests.
    """
    p_adf = adfuller(series, autolag='AIC')[1]
    with np.testing.suppress_warnings() as sup:
        sup.filter(InterpolationWarning)
        sup.filter(UserWarning)
        p_kpss = kpss(series, nlags="legacy")[1]
    return p_adf, p_kpss

def enforce_stationarity(
    series: np.ndarray,
    alpha_adf: float = 0.05,
    alpha_kpss: float = 0.05,
    diff_order: int = 1
) -> np.ndarray:
    """
    Enforce stationarity on a series by differencing if needed.
    
    Args:
        series (np.ndarray): Input time series.
        alpha_adf (float): Significance level for ADF test.
        alpha_kpss (float): Significance level for KPSS test.
        diff_order (int): Maximum order of differencing to apply.
        
    Returns:
        np.ndarray: Stationary time series.
    """
    if diff_order <= 0:
        logger.info("Differencing order is <= 0. No stationarity enforcement via differencing.")
        return series.astype(np.float32)
        
    current_series = series.astype(np.float32)
    p_adf, p_kpss = adf_kpss_test(current_series)
    
    if (p_adf >= alpha_adf) or (p_kpss <= alpha_kpss):
        logger.info(f"Series is non-stationary (ADF p={p_adf:.3f}, KPSS p={p_kpss:.3f}). Applying differencing of order {diff_order}.")
        effective_diff_order = min(diff_order, len(current_series) - 1)
        if effective_diff_order < diff_order:
            logger.warning(f"diff_order reduced from {diff_order} to {effective_diff_order} due to series length.")
        if effective_diff_order > 0:
            current_series = np.diff(current_series, n=effective_diff_order).astype(np.float32)
        else:
            logger.warning("Cannot apply differencing as effective_diff_order is 0 (series too short).")
    else:
        logger.info(f"Series is stationary (ADF p={p_adf:.3f}, KPSS p={p_kpss:.3f}). No differencing applied.")
    return current_series

def scale_array(
    arr: np.ndarray,
    method: Optional[str] = "minmax",
    fit_only_on: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Scale a NumPy array using specified sklearn preprocessor.
    
    Args:
        arr (np.ndarray): Array to scale.
        method (Optional[str]): Scaling method ('minmax', 'standard', 'power', 'quantile', or None).
        fit_only_on (Optional[np.ndarray]): If provided, scaler is fit on this array.
        **kwargs: Additional keyword arguments for the scaler constructor.
        
    Returns:
        Tuple[np.ndarray, Optional[object]]: Scaled array and fitted scaler object.
    """
    if method is None or method.lower() == 'none':
        return arr.astype(np.float32), None

    methods = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "power": PowerTransformer,
        "quantile": QuantileTransformer,
    }
    if method not in methods:
        raise ValueError(f"Unknown scaling method '{method}'. Supported: {list(methods.keys())} or None.")

    Scaler_cls = methods[method]
    scaler_instance = Scaler_cls(**kwargs)
    
    data_to_fit = fit_only_on if fit_only_on is not None else arr
    if data_to_fit.ndim == 1:
        data_to_fit = data_to_fit.reshape(-1, 1)
    
    try:
        scaler_instance.fit(data_to_fit)
    except ValueError as e:
        logger.warning(f"Scaler fitting failed for method '{method}' with error: {e}. Returning unscaled data.")
        return arr.astype(np.float32), None

    arr_reshaped = arr.reshape(-1, 1) if arr.ndim == 1 else arr
    
    try:
        scaled_arr = scaler_instance.transform(arr_reshaped)
    except ValueError as e:
        logger.warning(f"Scaler transformation failed for method '{method}' with error: {e}. Returning unscaled data.")
        return arr.astype(np.float32), scaler_instance

    if arr.ndim == 1 and scaled_arr.ndim == 2 and scaled_arr.shape[1] == 1:
        scaled_arr = scaled_arr.ravel()
        
    return scaled_arr.astype(np.float32), scaler_instance

def scale_series(series: np.ndarray, method: Optional[str] = "minmax", **kwargs) -> Tuple[np.ndarray, Optional[object]]:
    """
    Scale a 1D time series.
    
    Args:
        series (np.ndarray): 1D time series to scale.
        method (Optional[str]): Scaling method.
        **kwargs: Additional keyword arguments for the scaler.
        
    Returns:
        Tuple[np.ndarray, Optional[object]]: Scaled series and fitted scaler object.
    """
    if series.ndim != 1:
        logger.warning(f"scale_series expects a 1D array, got shape {series.shape}. Attempting to ravel.")
        series_1d = series.ravel()
    else:
        series_1d = series
    return scale_array(series_1d, method=method, **kwargs)

class FeatureEngineer:
    """
    Feature engineering for time series data.
    
    Args:
        lags (Optional[List[int]]): List of lag values to create.
        rolling_windows (Optional[List[int]]): List of window sizes for rolling statistics.
        cyclical_features (Optional[Dict[str, int]]): Dictionary of cyclical features to create.
    """
    def __init__(
        self,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None,
        cyclical_features: Optional[Dict[str, int]] = None 
    ):
        self.lags = sorted(list(set(lags))) if lags is not None else []
        self.rolling_windows = sorted(list(set(rolling_windows))) if rolling_windows is not None else []
        self.cyclical_features = cyclical_features if cyclical_features is not None else {}
        logger.info(f"FeatureEngineer initialized with lags={self.lags}, rolling_windows={self.rolling_windows}, cyclical_features={self.cyclical_features}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input DataFrame by adding engineered features.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'value' column and DateTimeIndex.
            
        Returns:
            pd.DataFrame: DataFrame with added features.
        """
        df_feat = df.copy()
        if "value" not in df_feat.columns:
            logger.error("FeatureEngineer expects a 'value' column in the input DataFrame.")
            return df_feat

        for lag in self.lags:
            if lag > 0:
                df_feat[f"lag_{lag}"] = df_feat["value"].shift(lag)

        for w in self.rolling_windows:
            if w > 0:
                shifted_values = df_feat["value"].shift(1)
                df_feat[f"roll_mean_{w}"] = shifted_values.rolling(window=w, min_periods=1).mean()
                df_feat[f"roll_std_{w}"] = shifted_values.rolling(window=w, min_periods=1).std()

        if isinstance(df_feat.index, pd.DatetimeIndex) and self.cyclical_features:
            for time_attr, period in self.cyclical_features.items():
                if hasattr(df_feat.index, time_attr):
                    values = getattr(df_feat.index, time_attr)
                    df_feat[f"{time_attr}_sin"] = np.sin(2 * np.pi * values / period)
                    df_feat[f"{time_attr}_cos"] = np.cos(2 * np.pi * values / period)
                else:
                    logger.warning(f"DatetimeIndex does not have attribute '{time_attr}' for cyclical feature generation.")
        
        return df_feat

def create_windows(
    data: np.ndarray,
    lookback: int,
    horizon: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create windowed X and y tensors for time series forecasting.
    
    Args:
        data (np.ndarray): Input data of shape (T, F_all_features).
        lookback (int): Number of time steps to look back.
        horizon (int): Number of steps to forecast ahead.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: X and y tensors for training.
        
    Raises:
        ValueError: If there's not enough data for the specified lookback and horizon.
    """
    if data.ndim == 1:
        logger.debug("create_windows received 1D data, reshaping to (T, 1).")
        data = data.reshape(-1, 1)

    T, F_all_features = data.shape
    N = T - lookback - horizon + 1
    
    if N <= 0:
        raise ValueError(
            f"Not enough data points (T={T}, F={F_all_features}) for lookback={lookback} + horizon={horizon}. "
            f"Need T >= lookback + horizon. (Currently T - lookback - horizon + 1 = {N})"
        )

    X = torch.zeros((N, lookback, F_all_features), dtype=torch.float32)
    y = torch.zeros((N, horizon), dtype=torch.float32)

    for i in range(N):
        X[i] = torch.from_numpy(data[i:i + lookback])
        y[i] = torch.from_numpy(data[i + lookback:i + lookback + horizon, 0])

    return X, y

def jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise to input tensor.
    
    Args:
        x (torch.Tensor): Input tensor.
        sigma (float): Standard deviation of the noise.
        
    Returns:
        torch.Tensor: Tensor with added noise.
    """
    return x + torch.randn_like(x) * sigma

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series forecasting.
    
    Args:
        series_source: Source of time series data (file path or array).
        lookback (int): Number of time steps to look back.
        horizon (int): Number of steps to forecast ahead.
        enforce_stat (bool): Whether to enforce stationarity.
        diff_order (int): Order of differencing for stationarity.
        scale_method (Optional[str]): Method for scaling the data.
        scale_kwargs (Optional[dict]): Additional arguments for scaling.
        engineer (Optional[FeatureEngineer]): Feature engineering instance.
        augment_fns (Optional[List[Callable]]): List of augmentation functions.
    """
    def __init__(
        self,
        series_source: Union[str, np.ndarray, pd.Series, pd.DataFrame],
        lookback: int,
        horizon: int = 1,
        *,
        enforce_stat: bool = False,
        diff_order: int = 1,
        scale_method: Optional[str] = "minmax",
        scale_kwargs: Optional[dict] = None,
        engineer: Optional[FeatureEngineer] = None,
        augment_fns: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.scale_method = scale_method
        self.scale_kwargs = scale_kwargs or {}
        self.engineer = engineer
        self.augment_fns = augment_fns or []

        if isinstance(series_source, str):
            series = load_series(series_source)
        elif isinstance(series_source, (pd.Series, pd.DataFrame)):
            series = series_source.values
        else:
            series = series_source

        if enforce_stat:
            series = enforce_stationarity(series, diff_order=diff_order)

        if scale_method:
            series, self.scaler = scale_series(series, method=scale_method, **self.scale_kwargs)
        else:
            self.scaler = None

        if engineer is not None:
            df = pd.DataFrame({'value': series})
            if isinstance(series_source, (pd.Series, pd.DataFrame)) and isinstance(series_source.index, pd.DatetimeIndex):
                df.index = series_source.index
            df = engineer.transform(df)
            series = df.values

        self.X, self.y = create_windows(series, lookback, horizon)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.X[idx], self.y[idx]
        for fn in self.augment_fns:
            x = fn(x)
        return x, y

