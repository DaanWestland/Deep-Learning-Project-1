# Time Series Forecasting with GRU

This project implements a deep learning-based time series forecasting system using Gated Recurrent Units (GRU). The system is designed for one-step ahead forecasting and includes comprehensive training, evaluation, and baseline comparison capabilities.

## Project Structure

```
.
├── data/               # Directory for storing time series data
├── src/               # Source code directory
│   ├── data_loader.py # Data loading and preprocessing utilities
│   ├── models.py      # Neural network model definitions
│   ├── train.py       # Training pipeline and hyperparameter optimization
│   ├── evaluate.py    # Model evaluation and metrics
│   └── baselines.py   # Baseline models for comparison
├── models/            # Directory for trained models and related files
│   ├── gru_tuned_best.pth        # Best performing model checkpoint
│   ├── gru_tuned_scaler.joblib   # Data scaler used for preprocessing
│   ├── gru_tuned_best_params.json # Best hyperparameters
│   └── gru_tuned_study.pkl       # Optuna study results
├── eval_results/      # Directory for evaluation outputs
│   ├── forecast.png   # Visualization of model predictions
│   ├── metrics.json   # Evaluation metrics (if test data provided)
│   └── gru_tuned_best_ts.pt     # TorchScript version of the model
├── baseline_results/  # Directory for baseline model results
└── README.md          # This file
```

## Features

- GRU-based time series forecasting model
- Hyperparameter optimization using Optuna
- K-fold cross-validation support
- Multiple baseline models for comparison
- Comprehensive evaluation metrics
- Data preprocessing and scaling utilities
- Model persistence and checkpointing
- TorchScript export for production deployment

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- Optuna
- Matplotlib
- joblib

## Usage

### Training

1. Place your time series data in the `data/` directory
2. Configure the training parameters in `src/train.py`:
   - Number of cross-validation trials
   - Number of folds
   - Training epochs
   - Early stopping patience
   - Validation split ratio

3. Run the training pipeline:
   ```bash
   python src/train.py
   ```

   This will:
   - Perform hyperparameter optimization using Optuna
   - Save the best model and hyperparameters
   - Store the scaler for data preprocessing
   - Save the complete Optuna study

### Evaluation

1. Evaluate the trained model:
   ```bash
   python src/evaluate.py --checkpoint models/gru_tuned_best.pth
   ```

   Optional arguments:
   - `--scaler`: Path to scaler file (default: inferred from checkpoint name)
   - `--hist-data`: Path to historical data (default: data/Xtrain.mat)
   - `--test-data`: Path to test data for metrics (optional)
   - `--n-steps`: Number of steps to forecast (default: 200)
   - `--results-dir`: Output directory (default: eval_results)
   - `--log-level`: Logging level (default: INFO)

## Model Files

The trained model and related files are stored in the `models/` directory:

- `gru_tuned_best.pth`: The best performing model checkpoint
  - Contains model state and hyperparameters
  - Can be loaded for inference or further training
- `gru_tuned_scaler.joblib`: The data scaler used for preprocessing
  - Used to normalize input data
  - Required for consistent preprocessing during inference
- `gru_tuned_best_params.json`: The optimal hyperparameters found during training
  - Includes model architecture parameters
  - Training configuration
  - Optimization settings
- `gru_tuned_study.pkl`: The complete Optuna study results
  - Contains all trial results
  - Can be used for analysis of hyperparameter optimization

## Evaluation Results

The evaluation process generates several outputs stored in the `eval_results/` directory:

- `forecast.png`: A visualization comparing the model's predictions with actual values
  - Shows historical data
  - Forecasted values
  - Ground truth (if test data provided)
- `metrics.json`: Evaluation metrics (if test data provided)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- `gru_tuned_best_ts.pt`: A TorchScript version of the model
  - Optimized for production deployment
  - Can be loaded without Python dependencies
  - Supports C++ deployment

## Model Architecture

The project implements two GRU-based forecasting models:

### GRUForecast
- Single-layer GRU with configurable hidden size
- Dropout for regularization
- Final linear layer for prediction
- Batch-first processing
- Configurable input and output dimensions
- Direct multi-step prediction

### GRUSeq2Seq
- Encoder-decoder architecture
- Teacher forcing support
- Iterative multi-step prediction
- Configurable horizon
- Dropout for regularization

## Evaluation

The system provides comprehensive evaluation capabilities including:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Comparison with baseline models
- Visualization of predictions
- Multi-step forecasting support
- Production-ready model export


