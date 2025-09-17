"""
Tests for WindowMixingDataset class.

This module tests the WindowMixingDataset functionality to ensure proper
behavior for windowed time series data mixing with single worker.
"""

import pytest
import torch
from torch.utils.data import IterableDataset
from unittest.mock import patch
from typing import List, Iterator

# Mock TensorIterableDataset since it's from an external package
class MockTensorIterableDataset(IterableDataset):
    """Mock implementation of TensorIterableDataset for testing."""
    
    def __init__(self, data: List[torch.Tensor], name: str = "mock"):
        super().__init__()
        self.data = data
        self.name = name
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        for item in self.data:
            yield item


class TestWindowMixingDataset:
    """Test the WindowMixingDataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mock windowed datasets with different data
        self.window_size = 32
        self.prediction_depth = 1
        self.batch_size = 4
        self.prefetch_depth = 8
        self.seed = 42
        
        # Create sample data for each dataset - each item should be a single tensor
        torch.manual_seed(42)
        data1 = [torch.randn(self.window_size + self.prediction_depth) for _ in range(20)]
        data2 = [torch.randn(self.window_size + self.prediction_depth) for _ in range(20)]
        data3 = [torch.randn(self.window_size + self.prediction_depth) for _ in range(20)]
        
        self.windowed_datasets = [
            MockTensorIterableDataset(data1, "dataset1"),
            MockTensorIterableDataset(data2, "dataset2"),
            MockTensorIterableDataset(data3, "dataset3"),
        ]

    @patch('binconvfm.utils.download.window_mixing_dataset.get_worker_info')
    def test_single_dataset(self, mock_get_worker_info):
        """Test behavior with single dataset."""
        from binconvfm.utils.download.window_mixing_dataset import WindowMixingDataset
        
        # Mock no worker info (single worker case)
        mock_get_worker_info.return_value = None
        
        # Create single dataset
        single_dataset = [self.windowed_datasets[0]]
        
        dataset = WindowMixingDataset(
            windowed_datasets=single_dataset,
            window_size=self.window_size,
            prediction_depth=self.prediction_depth,
            seed=self.seed,
            batch_size=self.batch_size,
            prefetch_depth=self.prefetch_depth
        )
        
        batches = list(dataset)
        assert len(batches) > 0
        
        # Check structure
        for batch in batches:
            X, y = batch
            assert X.shape == (self.batch_size, self.window_size)
            assert y.shape == (self.batch_size, self.prediction_depth)


if __name__ == "__main__":
    pytest.main([__file__])
