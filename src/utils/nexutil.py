import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chronos import ChronosPipeline
from nixtla import NixtlaClient
#from neuralforecast.models import MLP, NBEATS
#from neuralforecast.losses.pytorch import HuberLoss
#from neuralforecast.core import NeuralForecast
#from tqdm import tqdm
import pickle
import yaml
import argparse
import logging

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='NexData Loader')
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Increase verbosity level')
    return parser.parse_args()

def get_log_level(verbosity):
    """
    Get logging level based on verbosity.

    Args:
        verbosity (int): Verbosity level.

    Returns:
        int: Corresponding logging level.
    """
    if not isinstance(verbosity, int):
        raise ValueError("verbosity must be an integer")

    if verbosity == 0:
        return logging.WARNING
    elif verbosity == 1:
        return logging.INFO
    elif verbosity >= 2:
        return logging.DEBUG
    else:
        raise ValueError("verbosity must be a non-negative integer")

# Function to calculate the Index of Agreement (IoA)
def calculate_ioa(y_true, y_pred):
    """
    Calculate the Index of Agreement (IoA) between true and predicted values.

    Args:
        y_true (numpy.ndarray): True values.
        y_pred (numpy.ndarray): Predicted values.

    Returns:
        float: Index of Agreement (IoA) between true and predicted values.
    """
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((np.abs(y_pred - np.mean(y_true)) + \
                        np.abs(y_true - np.mean(y_true)))**2)
    ioa = 1 - (numerator / denominator)
    return ioa

# Function to plot the time series
def plot_time_series(y_true, y_pred):
    """
    Plot the true and predicted time series.

    Args:
        y_true (numpy.ndarray): True time series.
        y_pred (numpy.ndarray): Predicted time series.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='Real')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Hours')
    plt.ylabel('Current Velocity')
    plt.title('Time Series: Current Velocity')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_api_key(file_path):
    """
    Load API key from a file.

    Args:
        file_path (str): The path to the file containing the API key.

    Returns:
        str: The API key loaded from the file.
    """
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the API key from the file and strip any leading whitespace
        api_key = file.read().strip()
    return api_key

def normalize_z_score(series,
                      mean=None,
                      std=None):
    """
    Normalize a series using Z-score normalization.

    Parameters:
    series (pd.Series or list): The input series to normalize.

    Returns:
    normalized_series (pd.Series): The normalized series.
    mean (float): The mean of the original series.
    std (float): The standard deviation of the original series.
    """
    series = pd.Series(series)  # Convert to pd.Series if it is a list
    if not mean: mean = series.mean()
    if not std: mean = series.std()
    normalized_series = (series - mean) / std
    return normalized_series, mean, std

def denormalize_z_score(normalized_series, mean, std):
    """
    Denormalize a series using Z-score normalization parameters.

    Parameters:
    normalized_series (pd.Series or list): The normalized series to denormalize.
    mean (float): The mean used for the original normalization.
    std (float): The standard deviation used for the original normalization.

    Returns:
    denormalized_series (pd.Series): The denormalized series.
    """
    normalized_series = pd.Series(normalized_series)
    denormalized_series = (normalized_series * std) + mean
    return denormalized_series

import pandas as pd
from typing import Tuple, Dict, Optional

def normalize_z_score_df(df: pd.DataFrame, 
                         means: Optional[Dict[str, float]] = None, 
                         stds: Optional[Dict[str, float]] = None
                        ) -> Tuple[pd.DataFrame,
                                   Dict[str, float],
                                   Dict[str, float]]:
    """
    Normalize a DataFrame using Z-score normalization,
        excluding the datetime column.

    Parameters:
    df (pd.DataFrame): The input DataFrame to normalize.
    means (dict, optional): The means of the DataFrame columns.
        Calculated if not provided.
    stds (dict, optional): The standard deviations of the DataFrame columns.
        Calculated if not provided.

    Returns:
    Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]: 
        - The normalized DataFrame.
        - The means of the DataFrame columns.
        - The standard deviations of the DataFrame columns.
    """
    calc_means = means is None
    calc_stds = stds is None
    
    normalized_df = df.copy()
    
    if calc_means:
        means = {}
    if calc_stds:
        stds = {}
    
    for column in df.columns:
        if df[column].dtype != 'datetime64[ns]':
            if calc_means:
                means[column] = df[column].mean()
            if calc_stds:
                stds[column] = df[column].std()

            mean = means[column]
            std = stds[column]

            normalized_df[column] = (df[column] - mean) / std

    return normalized_df, means, stds


def denormalize_z_score_df(normalized_df, means, stds):
    """
    Denormalize a DataFrame using Z-score normalization parameters.

    Parameters:
    normalized_df (pd.DataFrame): The normalized DataFrame to denormalize.
    means (dict): The means used for the original normalization.
    stds (dict): The standard deviations used for the original normalization.

    Returns:
    denormalized_df (pd.DataFrame): The denormalized DataFrame.
    """
    denormalized_df = normalized_df.copy()
    
    for column in normalized_df.columns:
        if column in means and column in stds:
            denormalized_df[column] = (normalized_df[column] * \
                                       stds[column]) + means[column]

    return denormalized_df