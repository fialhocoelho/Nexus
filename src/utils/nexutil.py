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
        # Read the API key from the file and strip any leading/trailing whitespace
        api_key = file.read().strip()
    return api_key