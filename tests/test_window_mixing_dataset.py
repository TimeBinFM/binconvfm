"""
Tests for WindowMixingDataset class.

This module tests the WindowMixingDataset functionality to ensure proper
behavior for windowed time series data mixing with single worker.
"""

import pytest
import torch
from unittest.mock import patch
from typing import List, Iterator
from preprocessing.common import TensorIterableDataset

# Mock TensorIterableDataset since it's from an external package
class MockTensorIterableDataset(TensorIterableDataset):
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
        
        # Create sample data for each dataset - each item should be a 2D tensor (batch_size, window_size + prediction_depth)
        torch.manual_seed(42)
        data1 = [torch.randn(self.batch_size, self.window_size + self.prediction_depth) for _ in range(20)]
        
        self.windowed_dataset = MockTensorIterableDataset(data1, "dataset1")

    def test_single_dataset(self):
        """Test behavior with single dataset."""
        from binconvfm.utils.download.window_mixing_dataset import WindowMixingDataset
        
        # Create single dataset
        
        dataset = WindowMixingDataset(
            windowed_dataset=self.windowed_dataset,
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
