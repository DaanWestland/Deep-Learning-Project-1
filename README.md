# Time Series Forecasting with GRU

This project implements a deep learning-based time series forecasting system using Gated Recurrent Units (GRU). The system is designed for one-step ahead forecasting and includes comprehensive training, evaluation, and baseline comparison capabilities.

## Features

- **Advanced GRU Models**:
  - `GRUForecast`: Direct multi-step prediction model
  - `GRUSeq2Seq`: Encoder-decoder architecture with teacher forcing
  - Configurable architecture (layers, hidden size, dropout)

- **Data Processing**:
  - Automatic stationarity enforcement
  - Multiple scaling methods (MinMax, Standard, Power, Quantile)
  - Feature engineering (lags, rolling statistics, cyclical features)
  - Data augmentation support

- **Training Pipeline**:
  - Hyperparameter optimization using Optuna
  - K-fold cross-validation
  - Early stopping
  - Learning rate scheduling
  - Gradient clipping
  - Mixed precision training (AMP)

- **Evaluation**:
  - Comprehensive metrics (MAE, MSE, RMSE)
  - Visualization tools
  - Baseline model comparison
  - Multi-step forecasting support

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
│   ├── metrics.json   # Evaluation metrics
│   └── gru_tuned_best_ts.pt     # TorchScript version of the model
└── README.md          # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- NumPy
- pandas
- scikit-learn
- Optuna
- Matplotlib
- joblib
- statsmodels

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place your time series data in the `data/` directory
2. Supported formats:
   - `.mat` files with 'Xtrain', 'Xtest', or 'data' keys
   - NumPy arrays
   - Pandas Series/DataFrame
   - CSV files (will be converted to DataFrame)

### Training

1. Configure training parameters in `src/train.py`:
   ```python
   @dataclass
   class Config:
       data_path: str = 'data/Xtrain.mat'
       output_dir: str = 'models'
       base_name: str = 'gru'
       cv_trials: int = 500
       cv_folds: int = 10
       cv_epochs: int = 100
       cv_patience: int = 20
       final_epochs: int = 500
       final_patience: int = 100
       val_split: float = 0.1
   ```

2. Run the training pipeline:
   ```bash
   python src/train.py
   ```

   This will:
   - Perform hyperparameter optimization
   - Save the best model and hyperparameters
   - Store the data scaler
   - Save the Optuna study results

### Evaluation

1. Evaluate the trained model:
   ```bash
   python src/evaluate.py --checkpoint models/gru_tuned_best.pth
   ```

   Optional arguments:
   - `--scaler`: Path to scaler file
   - `--hist-data`: Path to historical data
   - `--test-data`: Path to test data for metrics
   - `--n-steps`: Number of steps to forecast
   - `--results-dir`: Output directory
   - `--log-level`: Logging level

2. View evaluation results in `eval_results/`:
   - `forecast.png`: Visualization of predictions
   - `metrics.json`: Evaluation metrics
   - `gru_tuned_best_ts.pt`: TorchScript model

## Model Architecture

### GRUForecast
- Single-layer GRU with configurable hidden size
- Dropout for regularization
- Final linear layer for prediction
- Direct multi-step prediction
- Batch-first processing

### GRUSeq2Seq
- Encoder-decoder architecture
- Teacher forcing support
- Iterative multi-step prediction
- Configurable horizon
- Dropout for regularization

## Hyperparameter Optimization

The system uses Optuna for hyperparameter optimization with the following search space:

- Model architecture:
  - Hidden size: 16-256
  - Number of layers: 1-6
  - Dropout: 0.0-0.5

- Training:
  - Learning rate: 1e-4 to 3e-2
  - Batch size: 8, 16, 32, 64
  - Optimizer: AdamW, Adam, RMSprop
  - Weight decay: 1e-7 to 1e-3

- Data processing:
  - Scaling method: minmax, standard, power, quantile
  - Stationarity enforcement
  - Feature engineering options

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


