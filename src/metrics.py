import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # For loading .mat files
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Configuration ---
TRAIN_DATA_PATH = 'data/Xtrain.mat'  # Path to your training data .mat file
TEST_DATA_PATH = 'data/Xtest.mat'    # Path to your actual test data .mat file (the 200 true values)

# Model 1 Configuration
MODEL1_NAME = 'GRUForecast'
PREDICTIONS_PATH_MODEL1 = 'gru_GRUForecast/predictions_actual.npy' # Path to Model 1's predictions

# Model 2 Configuration
MODEL2_NAME = 'GRUSeq2Seq'
PREDICTIONS_PATH_MODEL2 = 'gru_GRUSeq2Seq/predictions_actual.npy' # Path to Model 2's predictions


# Variable names within the .mat files (check your .mat files to confirm these)
# For example, if Xtrain.mat has a variable named 'data', set TRAIN_VAR_NAME = 'data'
TRAIN_VAR_NAME = 'Xtrain' # Common default, but might be 'y_train', 'train_data', etc.
TEST_VAR_NAME = 'Xtest'   # Common default, but might be 'y_test', 'test_data', etc.

# --- Helper Functions ---

def load_mat_data(file_path, var_name):
    """
    Loads data from a .mat file.

    Args:
        file_path (str): The path to the .mat file.
        var_name (str): The name of the variable to extract from the .mat file.

    Returns:
        numpy.ndarray: The data loaded from the .mat file, or None if an error occurs.
    """
    try:
        mat_contents = loadmat(file_path)
        data = mat_contents[var_name]
        # Ensure data is a 1D array (e.g., (n_samples,) )
        # .mat files often store vectors as (n_samples, 1) or (1, n_samples)
        if data.ndim > 1:
            data = data.squeeze() # Removes single-dimensional entries from the shape
        print(f"Successfully loaded '{var_name}' from {file_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except KeyError:
        print(f"Error: Variable '{var_name}' not found in {file_path}.")
        print(f"Available keys in {file_path}: {list(loadmat(file_path).keys())}")
        return None
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")
        return None

def load_npy_predictions(file_path, model_name="Model"):
    """
    Loads predictions from a .npy file.

    Args:
        file_path (str): The path to the .npy file.
        model_name (str): Name of the model for logging purposes.

    Returns:
        numpy.ndarray: The predictions loaded from the .npy file, or None if an error occurs.
    """
    try:
        predictions = np.load(file_path)
        # Ensure predictions are a 1D array
        if predictions.ndim > 1:
            predictions = predictions.squeeze()
        print(f"Successfully loaded predictions for {model_name} from {file_path}. Shape: {predictions.shape}")
        return predictions
    except FileNotFoundError:
        print(f"Error: Predictions file for {model_name} not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading predictions for {model_name} from {file_path}: {e}")
        return None

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Model Comparison Script ---")

    # 1. Load Training Data
    print("\n--- Loading Training Data ---")
    train_data = load_mat_data(TRAIN_DATA_PATH, TRAIN_VAR_NAME)
    if train_data is not None:
        print(f"Length of training data: {len(train_data)}")

    # 2. Load Actual Test Data
    print("\n--- Loading Actual Test Data ---")
    actual_test_data = load_mat_data(TEST_DATA_PATH, TEST_VAR_NAME)

    if actual_test_data is None:
        print("Halting script as actual test data could not be loaded.")
        exit()

    # Validate test data length (expecting 200 points as per assignment)
    if len(actual_test_data) != 200:
        print(f"Warning: Loaded actual test data has {len(actual_test_data)} points, but assignment specifies 200 for recursive prediction evaluation.")
        print("Proceeding with the loaded data, but please verify this is correct.")

    # 3. Load Model Predictions
    print("\n--- Loading Model Predictions ---")
    predictions_model1 = load_npy_predictions(PREDICTIONS_PATH_MODEL1, MODEL1_NAME)
    predictions_model2 = load_npy_predictions(PREDICTIONS_PATH_MODEL2, MODEL2_NAME)

    if predictions_model1 is None or predictions_model2 is None:
        print("Halting script as one or both model predictions could not be loaded.")
        exit()

    # 4. Validate Prediction Lengths
    print("\n--- Validating Prediction Lengths ---")
    valid_lengths = True
    for model_name, preds in [(MODEL1_NAME, predictions_model1), (MODEL2_NAME, predictions_model2)]:
        if len(preds) != len(actual_test_data):
            print(f"Error: Mismatch in length for {model_name}. Actual: {len(actual_test_data)}, Predicted: {len(preds)}.")
            valid_lengths = False
    
    if not valid_lengths:
        print("Please ensure your .npy files contain predictions for the correct number of test points.")
        exit()
    
    print(f"All predictions and actual test data have matching lengths: {len(actual_test_data)} points.")

    # 5. Calculate and Report Metrics for each model
    print("\n--- Calculating Performance Metrics ---")
    metrics_results = {}

    for model_name, predictions in [(MODEL1_NAME, predictions_model1), (MODEL2_NAME, predictions_model2)]:
        mae = mean_absolute_error(actual_test_data, predictions)
        mse = mean_squared_error(actual_test_data, predictions)
        rmse = np.sqrt(mse)
        metrics_results[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
        print(f"\nMetrics for {model_name}:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

    # 6. Create Comparison Plot
    print("\n--- Generating Comparison Plot ---")
    plt.figure(figsize=(15, 8)) # Slightly wider for better legend placement
    
    time_steps = np.arange(len(actual_test_data))

    plt.plot(time_steps, actual_test_data, label='Actual Values', color='black', linestyle='-', marker='o', markersize=4, linewidth=2)
    plt.plot(time_steps, predictions_model1, label=f'{MODEL1_NAME} Predictions', color='dodgerblue', linestyle='--', marker='x', markersize=4)
    plt.plot(time_steps, predictions_model2, label=f'{MODEL2_NAME} Predictions', color='red', linestyle=':', marker='s', markersize=4)
    
    plt.title('Comparison of Model Predictions vs. Actual Values', fontsize=16)
    plt.xlabel('Time Step (within the 200 point test sequence)', fontsize=12)
    plt.ylabel('Laser Measurement Value', fontsize=12)
    plt.legend(fontsize=10, loc='best') # 'best' tries to find the least obstructive location
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plot_filename = 'model_comparison_predictions_vs_actual.png'
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
        
    plt.show()

    print("\n--- Evaluation Script Finished ---")
