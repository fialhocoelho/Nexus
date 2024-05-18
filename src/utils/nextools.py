import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from chronos import ChronosPipeline
from nixtlats import NixtlaClient
from neuralforecast.models import MLP, NBEATS
from neuralforecast.losses.pytorch import HuberLoss
from neuralforecast.core import NeuralForecast
from tqdm import tqdm
import pickle
import yaml

def load_yaml_config(file_path):
    """
    Load configuration parameters from a YAML file.

    Parameters:
    file_path (str): Path to the YAML file containing configuration parameters.

    Returns:
    dict: A dictionary containing the configuration parameters.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to load data from a pickle file
def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Function to create input and output sequences
# def create_sequences(raw_data, context_len, forecast_len):
#    """Create input and output sequences from raw data."""
#    return create_input_output_sequences(raw_data, context_len, forecast_len)

# Function to create input-output sequences for training
def create_sequences(data, context_window_len, forecast_len):
    """
    Create input-output sequences for training.

    Args:
        data (numpy.ndarray): Time series data.
        context_window_len (int): len of the input sequence.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Input-output sequences.
    """
    X, y = [], []

    # Leaving space to predict forecast_len points in the future
    for i in range(len(data) - context_window_len - forecast_len +1):
        X.append(data[i:i+context_window_len])  # Input sequence
        # Output sequence (predict the next forecast_len points)
        y.append(data[i+context_window_len:i+context_window_len+forecast_len]) 
    return np.array(X), np.array(y)

# Function to split data into training and test sets
def split_data(data, split_idx):
    """Split data into training and test sets."""
    return data[:split_idx], data[split_idx:]

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