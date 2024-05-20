import os
import yaml
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset

# Function to load data from a pickle file
def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_yaml_config(file_path):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    file_path (str): Path to the YAML file containing configuration parameters.

    Returns:
    dict: A dictionary containing the configuration parameters.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Parameters:
    seed (int): The seed value to be set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_processed_data(processed_dir, train_filename, test_filename):
    """
    Load processed train and test dataframes.

    Parameters:
    processed_dir (str): Directory containing processed data.
    train_filename (str): Filename of the processed training data.
    test_filename (str): Filename of the processed test data.

    Returns:
    tuple: (pd.DataFrame, pd.DataFrame) Processed training and test data.
    """
    train_path = os.path.join(processed_dir, train_filename)
    test_path = os.path.join(processed_dir, test_filename)

    return pd.read_csv(train_path), pd.read_csv(test_path)

def create_sequences(series, context_window_len, forecast_len):
    """
    Create input and output sequences for training and testing.

    Parameters:
    series (np.ndarray): The time series data.
    context_window_len (int): The length of the input sequence.
    forecast_len (int): The length of the forecast (output) sequence.

    Returns:
    tuple: (np.ndarray, np.ndarray) Input and output sequences.
    """
    X, y = [], []
    for i in range(len(series) - context_window_len - forecast_len + 1):
        X.append(series[i:(i + context_window_len)])
        y.append(series[(i + context_window_len):(i + context_window_len + forecast_len)])
    return np.array(X), np.array(y)

def split_data(data, split_index):
    """
    Split data into training and test sets.

    Parameters:
    data (np.ndarray): The data to be split.
    split_index (int): The index at which to split the data.

    Returns:
    tuple: (np.ndarray, np.ndarray) Training and test data.
    """
    return data[:split_index], data[split_index:]

def load_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
    file_path (str): Path to the pickle file.

    Returns:
    any: Data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def transform_data(config_path,
                   to_root_dir=None,
                   teacher_rule='mean'):
    """
    Transform data using configuration parameters from a YAML file.

    Parameters:
    config_path (str): Path to the YAML file containing configuration parameters.

    Returns:
    tuple: Containing numpy arrays of train and test data.
    """
    # Load the configuration parameters from the YAML file
    config = load_yaml_config(config_path)
    data_params = config['data']
    model_params = config['model']

    # Set seeds for reproducibility
    set_random_seeds(data_params['default_seed'])

    # Define paths
    processed_dir = os.path.join(to_root_dir, data_params['processed_path'])
    interim_dir = os.path.join(to_root_dir, data_params['intermediate_path'])

    # Load dataframes from source
    train_df, test_df = load_processed_data(processed_dir, data_params['processed_train_df'], data_params['processed_test_df'])

    # Crop target datetime
    crop_datetime = data_params['crop_target_datetime']
    target_index = train_df.loc[train_df.ds == crop_datetime, 'ds'].index.values[0]

    # Processed series
    processed_series = np.concatenate([train_df[target_index:].y.values, test_df.y.values], axis=None)

    # Load forecast data
    timegpt_forecast_path = os.path.join(interim_dir, data_params['timegpt_fcst_file'])
    y_timegpt = np.array(load_pickle(timegpt_forecast_path))

    chronos_forecast_path = os.path.join(interim_dir, data_params['chronos_fcst_file'])
    y_chronos = np.array(load_pickle(chronos_forecast_path)).reshape((-1, y_timegpt.shape[1]))

    # Create input and output sequences
    X, y = create_sequences(processed_series, model_params['context_window_len'], model_params['forecast_len'])

    # Calculate the index to split data into training and test sets
    split_index = train_df.shape[0] - target_index - (model_params['context_window_len'] + model_params['forecast_len'] - 1)

    # Split input and output sequences into training and test sets
    X_train, X_test = split_data(X, split_index)
    y_train, y_test = split_data(y, split_index)

    # Split teacher features (foundation models prediction) into training and test sets
    y_train_timegpt, y_test_timegpt = split_data(y_timegpt, split_index)
    y_train_chronos, y_test_chronos = split_data(y_chronos, split_index)

    if teacher_rule=='chronos':
        y_teacher = y_chronos  # Choose Chronos as teacher features

    if teacher_rule=='timegpt':
        y_teacher = y_timegpt  # Choose TimeGPT-1 as teacher features

    if teacher_rule=='mean':
        y_teacher = np.mean([y_timegpt, y_chronos], axis=0) # Choose avg of foundation features as teacher features

    y_train_teacher, y_test_teacher = split_data(y_teacher, split_index)

    return X, y, X_train, X_test, y_train, y_test, y_train_teacher, y_test_teacher, y_train_timegpt, y_test_timegpt, y_train_chronos, y_test_chronos

def convert_to_tensors(X_train, X_test, y_train, y_test, y_train_teacher, y_test_teacher, device):
    """
    Convert numpy arrays to PyTorch tensors and move them to the specified device.

    Parameters:
    X_train (np.ndarray): Training input sequences.
    X_test (np.ndarray): Test input sequences.
    y_train (np.ndarray): Training output sequences.
    y_test (np.ndarray): Test output sequences.
    y_train_teacher (np.ndarray): Training teacher features.
    y_test_teacher (np.ndarray): Test teacher features.
    device (torch.device): The device to move the tensors to.

    Returns:
    tuple: Containing PyTorch tensors for train and test data.
    """
    to_tensor = lambda x: torch.from_numpy(x).float().to(device)
    X_train_tensor = to_tensor(X_train).unsqueeze(-1)
    X_test_tensor = to_tensor(X_test).unsqueeze(-1)
    y_train_tensor = to_tensor(y_train)
    y_test_tensor = to_tensor(y_test)
    y_train_teacher_tensor = to_tensor(y_train_teacher)
    y_test_teacher_tensor = to_tensor(y_test_teacher)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_train_teacher_tensor, y_test_teacher_tensor
