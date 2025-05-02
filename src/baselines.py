"""
This module implements baseline forecasting methods for time series prediction,
including ARIMA and Holt-Winters exponential smoothing. It provides utilities
for model fitting, forecasting, evaluation, and visualization of results.
"""

import os
import logging
import json

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_loader import load_series

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Forecasting functions
# -----------------------------------------------------------------------------

def arima_forecast(series, steps, order=(5, 1, 0)):
    """
    Fit an ARIMA model and forecast future values.
    
    ARIMA (AutoRegressive Integrated Moving Average) is a classic time series
    forecasting model that combines three components:
    1. Autoregression (AR): Uses past values to predict future values
    2. Integration (I): Uses differencing to make the time series stationary
    3. Moving Average (MA): Uses past forecast errors to predict future values
    
    Args:
        series: Time series data to fit, should be a 1D array
        steps: Number of steps to forecast into the future
        order: Tuple (p,d,q) where:
            p: order of autoregressive part (number of lag observations)
            d: degree of differencing (number of times data is differenced)
            q: order of moving average part (size of moving average window)
            
    Returns:
        np.ndarray: Array of forecasted values with length equal to 'steps'
    """
    logger.info(f"Fitting ARIMA{order}...")
    model = ARIMA(series, order=order).fit()
    logger.info("ARIMA fit complete. Forecasting...")
    return model.forecast(steps)


def holt_winters_forecast(series, steps, seasonal=None, seasonal_periods=None):
    """
    Fit Holt-Winters exponential smoothing and forecast future values.
    
    Holt-Winters is a forecasting method that uses exponential smoothing to
    capture three components of time series data:
    1. Level: The average value in the series
    2. Trend: The increasing or decreasing value in the series
    3. Seasonality: The repeating short-term cycle in the series
    
    Args:
        series: Time series data to fit, should be a 1D array
        steps: Number of steps to forecast into the future
        seasonal: Type of seasonality:
            - 'add': Additive seasonality (constant seasonal variations)
            - 'mul': Multiplicative seasonality (increasing seasonal variations)
            - None: No seasonality
        seasonal_periods: Number of periods in a seasonal cycle
            (e.g., 12 for monthly data with yearly seasonality)
        
    Returns:
        np.ndarray: Array of forecasted values with length equal to 'steps'
    """
    logger.info("Fitting Holt-Winters ExponentialSmoothing...")
    model = ExponentialSmoothing(
        series,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    ).fit()
    logger.info("Holt-Winters fit complete. Forecasting...")
    return model.forecast(steps)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def compute_metrics(true, pred):
    """
    Calculate evaluation metrics for predictions.
    
    This function computes common metrics for evaluating time series forecasts:
    - MAE (Mean Absolute Error): Average absolute difference between predictions and true values
    - MSE (Mean Squared Error): Average squared difference between predictions and true values
    - RMSE (Root Mean Squared Error): Square root of MSE, in the same units as the data
    
    Args:
        true: Ground truth values (1D array)
        pred: Predicted values (1D array)
        
    Returns:
        dict: Dictionary containing:
            - 'MAE': Mean Absolute Error
            - 'MSE': Mean Squared Error
            - 'RMSE': Root Mean Squared Error
    """
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


def plot_results(history, forecasts: dict, steps, save_dir=None):
    """
    Create a comparison plot of historical data and multiple forecast methods.
    
    This function generates a visualization that shows:
    1. Historical data in black
    2. Forecasts from different methods in different colors
    3. Clear labels and grid for easy interpretation
    
    Args:
        history: Historical time series data (1D array)
        forecasts: Dictionary mapping method names to their forecast arrays
        steps: Number of forecast steps
        save_dir: Optional directory to save the plot. If provided, the plot
                 will be saved as 'baseline_forecasts.png' in this directory
    """
    t_hist = np.arange(len(history))
    t_fut = np.arange(len(history), len(history) + steps)

    plt.figure(figsize=(12, 6))
    plt.plot(t_hist, history, label='Historical', color='black')

    for label, preds in forecasts.items():
        plt.plot(t_fut, preds, label=label)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f"Forecast Comparison ({steps} steps ahead)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'baseline_forecasts.png')
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Plot saved to {plot_path}")

    plt.show()

# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------

def run_baselines(
    data_path='data/Xtrain.mat',
    steps=200,
    arima_order=(5, 1, 0),
    hw_seasonal=None,
    hw_periods=None,
    results_dir='baseline_results'
):
    """
    Run baseline forecasting methods and compare their performance.
    
    This function serves as the main entry point for running baseline forecasting
    experiments. It:
    1. Loads the time series data from a .mat file
    2. Fits and forecasts using both ARIMA and Holt-Winters methods
    3. Attempts to load ground truth data for evaluation
    4. Computes and saves evaluation metrics if ground truth is available
    5. Creates and saves a comparison plot of the forecasts
    
    Args:
        data_path: Path to the training data .mat file
        steps: Number of steps to forecast into the future
        arima_order: Tuple (p,d,q) specifying the ARIMA model order
        hw_seasonal: Type of seasonality for Holt-Winters ('add', 'mul', or None)
        hw_periods: Number of periods in a seasonal cycle for Holt-Winters
        results_dir: Directory to save results and plots
        
    Returns:
        None
    """
    # Load training data
    logger.info(f"Loading series from {data_path}")
    series = load_series(data_path)

    # Generate forecasts using both methods
    arima_pred = arima_forecast(series, steps, order=arima_order)
    hw_pred = holt_winters_forecast(series, steps, seasonal=hw_seasonal, seasonal_periods=hw_periods)

    # Try to load ground truth for evaluation
    truth = None
    truth_path = data_path.replace('Xtrain', 'Xtest')
    if os.path.exists(truth_path):
        try:
            logger.info(f"Loading ground truth from {truth_path}")
            full_test = load_series(truth_path)
            truth = full_test[:steps]
        except Exception as e:
            logger.warning(f"Could not load ground truth: {e}")

    # Compute and save metrics if ground truth is available
    metrics = {}
    if truth is not None:
        metrics['ARIMA'] = compute_metrics(truth, arima_pred)
        metrics['HoltWinters'] = compute_metrics(truth, hw_pred)
        os.makedirs(results_dir, exist_ok=True)
        metrics_path = os.path.join(results_dir, 'baseline_metrics.json')
        with open(metrics_path, 'w') as mf:
            json.dump(metrics, mf, indent=4)
        logger.info(f"Metrics saved to {metrics_path}: {metrics}")

    # Create and save comparison plot
    forecasts = {'ARIMA': arima_pred, 'Holt-Winters': hw_pred}
    plot_results(series, forecasts, steps, save_dir=results_dir)

    logger.info("Baseline evaluation complete.")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_baselines()
