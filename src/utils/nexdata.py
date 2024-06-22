import os
import yaml
import pandas as pd
import numpy as np
import pickle
import torch
import logging
from tqdm import tqdm

from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# after neurips

import os
import logging
from datetime import datetime
import torch

class NexData:
    """
    A class to handle data and configuration for the Nexus project.

    Attributes:
    - nexus_folder (str): Path to the Nexus project folder.
    - config_path (str): Path to the configuration YAML file.
    - logger (logging.Logger): Logger instance for the class.
    - timestamp (str): Timestamp of the instance creation.
    - config (dict): Loaded configuration parameters.
    - data_params (dict): Data-related parameters from the config.
    - model_params (dict): Model-related parameters from the config.
    - features (dict): Feature-related parameters from the config.
    - device (torch.device): Torch device to be used for computations.
    - raw_dir (str): Path to the raw data directory.
    - processed_dir (str): Path to the processed data directory.
    - interim_dir (str): Path to the intermediate data directory.
    - train_folder_path (str): Path to the training data folder.
    - test_folder_path (str): Path to the testing data folder.
    """
    
    def __init__(self, nexus_folder='.',
                 config_path='config/config.yaml',
                 log_level=logging.INFO):
        """
        Initialize the NexData class.

        Parameters:
        - nexus_folder (str): Path to the Nexus project folder.
            Default is '.'.
        - config_path (str): Path to the configuration YAML file.
            Default is 'config/config.yaml'.
        - log_level (int): Logging level to use for the logger.
            Default is logging.INFO.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load the configuration parameters from the YAML file
        if not isinstance(config_path, str):
            raise ValueError("config_path must be a string")
        
        if config_path == 'config/config.yaml':
            config_path = os.path.join(nexus_folder, config_path)
        
        self.config = load_yaml_config(config_path)
        self.config_path = config_path
        self.logger.debug(f" Config loaded from: {self.config_path}")

        self.data_params = self.config['data']
        self.model_params = self.config['model']
        self.features = self.config['features']

        # Process each feature
        for feature in self.features:
            train_filepath = os.path.join(
                nexus_folder,
                self.data_params['raw_path'],
                self.data_params['train_folder'],
                self.features[feature]['train_filename'])
            self.features[feature]['train_filepath'] = train_filepath
            self.logger.debug(f' {feature} train path: {train_filepath}')

            test_filepath = os.path.join(
                nexus_folder,
                self.data_params['raw_path'],
                self.data_params['test_folder'],
                self.features[feature]['test_filename'])
            self.features[feature]['test_filepath'] = test_filepath
            self.logger.debug(f' {feature} test path: {test_filepath}')

        # Set seeds for reproducibility
        if 'default_seed' in self.data_params:
            set_random_seeds(self.data_params['default_seed'])
            self.logger.debug(f' Rnd seed: {self.data_params["default_seed"]}')
        else:
            self.logger.warning("No default seed found in data parameters")

        # Define the device
        try:
            self.device = torch.device(self.model_params['device'])
            self.logger.debug(f' Default device: {self.model_params["device"]}')
        except Exception as e:
            self.logger.error(f" An error occurred when setting device: {e}")
            raise

        # Define paths
        self.logger.info(' Defining paths...')
        self.raw_dir = os.path.join(nexus_folder, self.data_params['raw_path'])
        self.processed_dir = os.path.join(nexus_folder,
                                        self.data_params['processed_path'])
        self.interim_dir = os.path.join(nexus_folder,
                                        self.data_params['intermediate_path'])
        self.forecasted_dir = os.path.join(nexus_folder,
                                        self.data_params['forecasted_path'])
        self.train_folder_path = os.path.join(self.raw_dir,
                                            self.data_params['train_folder'])
        self.test_folder_path = os.path.join(self.raw_dir,
                                            self.data_params['test_folder'])
        self.nixtla_api_key_file = os.path.join(nexus_folder,
            self.model_params['nixtla_api_key_file'])

        self.logger.debug(' Paths are definied.')


def process_dataframe(df_source,
                        start_date,
                        end_date,
                        freq='1h',
                        interp_method=None,
                        datetime_col='datetime',
                        round_freq='5min'):
    """
    Process a dataframe to fill gaps in the datetime column and interpolate
    missing values.

    Parameters:
    df_source (pd.DataFrame): Source dataframe.
    start_date (str or pd.Timestamp): Start date for the date range.
    end_date (str or pd.Timestamp): End date for the date range.
    freq (str): Frequency for the new date range.
        Default is '1h'.
    interp_method (str, optional): Interpolation method.
        Default is None.
    datetime_col (str): Name of the datetime column.
        Default is 'datetime'.
    round_freq (str): Frequency to round the datetime values.
        Default is '5min'.

    Returns:
    pd.DataFrame: Processed dataframe with interpolated values.
    """
    if not isinstance(df_source, pd.DataFrame):
        raise ValueError("df_source must be a pandas DataFrame")
    if not isinstance(start_date, (str, pd.Timestamp)):
        raise ValueError("start_date must be a string or pandas Timestamp")
    if not isinstance(end_date, (str, pd.Timestamp)):
        raise ValueError("end_date must be a string or pandas Timestamp")
    if not isinstance(freq, str):
        raise ValueError("freq must be a string")
    if interp_method and interp_method not in [
        'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear',
        'quadratic', 'cubic', 'barycentric', 'krogh', 'polynomial', 'spline',
        'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'
    ]:
        intrp_msg = 'interp_method must be a valid interp. method or None'
        raise ValueError(intrp_msg)
    if not isinstance(datetime_col, str):
        raise ValueError("datetime_col must be a string")
    if not isinstance(round_freq, str):
        raise ValueError("round_freq must be a string")

    df_original = df_source.copy()

    # Remove timezone from the original dataframe
    df_original[datetime_col] = pd.to_datetime(
        df_original[datetime_col]).dt.tz_localize(None)

    # Create a DataFrame with the new date range
    df_processed = pd.DataFrame({datetime_col: pd.date_range(start=start_date,
                                                            end=end_date,
                                                            freq=freq)})

    # Round datetime values to match existing values in the original dataframe
    df_original[datetime_col] = df_original[datetime_col].dt.round(round_freq)

    # Merge original and processed dataframes to fill gaps in datetime
    merged_df = pd.merge(df_processed, df_original,
                        how='left', on=datetime_col)

    # Interpolate missing values if interpolation method is provided
    if interp_method:
        try:
            interpolated_df = merged_df.interpolate(method=interp_method,
                                                    limit_direction='both')
        except ValueError as e:
            raise ValueError(f"Interpolation failed: {e}")
    else:
        interpolated_df = merged_df

    final_df = interpolated_df.drop_duplicates(
        subset=datetime_col, keep='first').reset_index(drop=True)

    # Check for NaN values in columns other than 'datetime'
    check_isna = final_df.drop(columns=[datetime_col]).isnull().values.any()
    if check_isna and interp_method:
        raise ValueError("The df contains NaN values in cols.")

    return final_df

def generate_indices(df, context_len, forecast_len, shift, mode="sliding"):
    """
    Generate indices for context and forecast windows from a dataframe.

    Parameters:
    df (pd.DataFrame): Source dataframe.
    context_len (int): Length of the context window.
    forecast_len (int): Length of the forecast window.
    shift (int): Shift step to slide the window.
    mode (str): Mode for generating indices,
        either "sliding" or "fixed".

    Returns:
    list: Indices for context windows.
    list: Indices for forecast windows.
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not isinstance(context_len, int) or context_len <= 0:
        raise ValueError("context_len must be a positive integer")
    if not isinstance(forecast_len, int) or forecast_len <= 0:
        raise ValueError("forecast_len must be a positive integer")
    if not isinstance(shift, int) or shift <= 0:
        raise ValueError("shift must be a positive integer")
    if mode not in ["sliding", "fixed"]:
        raise ValueError("mode must be either 'sliding' or 'fixed'")

    # Calculate the total number of windows that can be generated
    range_size = int((df.shape[0] - context_len - forecast_len) / shift) + 1
    
    # Initialize lists to store the indices for context and forecast windows
    X_index = []
    y_index = []

    # Loop to generate indices for each window
    for i in range(range_size):
        X_start = shift * i  # Start index for the context window
        X_end = X_start + context_len  # End index for the context window
        y_start = X_end  # Start index for the forecast window
        y_end = y_start + forecast_len  # End index for the forecast window

        # Append the indices for the context and forecast windows
        if mode == 'sliding':
            X_index.append(df[X_start:X_end].index)
        if mode == 'fixed':
            X_index.append(df[0:X_end].index)
        y_index.append(df[y_start:y_end].index)

    return X_index, y_index

# Function to load data from a pickle file
def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_yaml_config(file_path):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    file_path (str): Path to the YAML file containing
        configuration parameters.

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


################################################################################
################################################################################
####################                                        ####################
####################    Functions before Neurips dataset    ####################
####################                                        ####################
################################################################################
################################################################################

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

'''
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
        y.append(series[(i + context_window_len):\
            (i + context_window_len + forecast_len)])
    return np.array(X), np.array(y)
'''
    
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
                   nexus_folder=None,
                   teacher_rule='mean'):
    """
    Transform data using configuration parameters from a YAML file.

    Parameters:
    config_path (str): Path to the YAML file containing
        configuration parameters.

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
    processed_dir = os.path.join(nexus_folder, data_params['processed_path'])
    interim_dir = os.path.join(nexus_folder, data_params['intermediate_path'])

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
