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

## Requirements

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- Optuna
- Matplotlib
- joblib

## Usage

1. Place your time series data in the `data/` directory
2. Configure the training parameters in `src/train.py`
3. Run the training pipeline:
   ```bash
   python src/train.py
   ```
4. Evaluate the trained model:
   ```bash
   python src/evaluate.py --checkpoint models/gru_tuned_best.pth
   ```

## Model Files

The trained model and related files are stored in the `models/` directory:
- `gru_tuned_best.pth`: The best performing model checkpoint
- `gru_tuned_scaler.joblib`: The data scaler used for preprocessing
- `gru_tuned_best_params.json`: The optimal hyperparameters found during training
- `gru_tuned_study.pkl`: The complete Optuna study results

## Evaluation Results

The evaluation process generates several outputs stored in the `eval_results/` directory:
- `forecast.png`: A visualization comparing the model's predictions with actual values
- `gru_tuned_best_ts.pt`: A TorchScript version of the model for production deployment

## Model Architecture

The project implements a GRU-based forecasting model with the following features:
- Single-layer GRU with configurable hidden size
- Dropout for regularization
- Final linear layer for prediction
- Batch-first processing
- Configurable input and output dimensions

## Evaluation

The system provides comprehensive evaluation capabilities including:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Comparison with baseline models
- Visualization of predictions


