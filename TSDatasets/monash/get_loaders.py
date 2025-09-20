"""Monash dataset loaders for time series forecasting.

This module provides utilities for loading and processing Monash time series datasets
into PyTorch-compatible data loaders with sliding window transformations.
"""

import torch
from torch.utils.data import IterableDataset
import numpy as np
from typing import Tuple, Iterator, Union, List

from .utils import load_train_test
from preprocessing.transform.dataset_builder import Builder


class TimeSeriesIterableDataset(IterableDataset):
    """An iterable dataset for time series data.
    
    This dataset converts numpy arrays to PyTorch tensors on-the-fly,
    making it memory efficient for large time series datasets.
    
    Args:
        series_array: Array-like object containing time series data.
                     Can be a NumpyExtensionArray or list of numpy arrays.
    
    Yields:
        torch.Tensor: Individual time series as float32 tensors.
    """
    
    def __init__(self, series_array: Union[np.ndarray, List[np.ndarray]]) -> None:
        # Convert to a Python list of numpy arrays
        self.series_list = list(series_array)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over time series, yielding PyTorch tensors."""
        for arr in self.series_list:
            yield torch.as_tensor(arr, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Return the number of time series in the dataset."""
        return len(self.series_list)


def get_loaders(
    context_length: int,
    prediction_depth_train: int,
    prediction_depth_test: int,
    filename: str = "monash_data/tourism_monthly_dataset.tsf",
    verbose: bool = False
) -> Tuple[IterableDataset, IterableDataset]:
    """Create training and testing data loaders for Monash time series datasets.
    
    This function loads time series data from a TSF file and creates PyTorch-compatible
    data loaders with sliding window transformations. Each sample consists of a context
    window and a prediction target.
    
    Args:
        context_length: Length of the input context window.
        prediction_depth_train: Number of future time steps to predict for training.
        prediction_depth_test: Number of future time steps to predict for testing.
        filename: Path to the TSF file containing the time series data.
        verbose: If True, print detailed information about the loaded dataset.
    
    Returns:
        A tuple containing:
        - Training dataset: IterableDataset yielding (context, target) pairs
        - Testing dataset: IterableDataset yielding (context, target) pairs
    
    Example:
        >>> train_loader, test_loader = get_loaders(
        ...     context_length=24,
        ...     prediction_depth_train=12,
        ...     prediction_depth_test=12,
        ...     filename="data/tourism.tsf"
        ... )
    """
    # Load and preprocess the data
    loaded_data = load_train_test(
        context_length=context_length,
        prediction_depth_train=prediction_depth_train,
        prediction_depth_test=prediction_depth_test,
        filename=filename,
        verbose=verbose
    )
    
    # Create training dataset with sliding windows
    train_window_size = context_length + prediction_depth_train
    ds_train = (
        Builder(TimeSeriesIterableDataset(loaded_data['train']))
        .sliding_window(train_window_size)
        .map(lambda t: (t[:context_length].unsqueeze(-1), t[context_length:].unsqueeze(-1)))
        .build()
    )
    
    # Create testing dataset with sliding windows  
    test_window_size = context_length + prediction_depth_test
    ds_test = (
        Builder(TimeSeriesIterableDataset(loaded_data['test']))
        .sliding_window(test_window_size)
        .map(lambda t: (t[:context_length].unsqueeze(-1), t[context_length:].unsqueeze(-1)))
        .build()
    )
    
    return ds_train, ds_test
