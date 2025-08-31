"""Tests for linear trend synthetic dataset."""

import torch
import pytest
from preprocessing.synthetic.linear_trend import LinearTrendDataset


def test_linear_trend_dataset_basic():
    """Test basic functionality of LinearTrendDataset."""
    dataset = LinearTrendDataset(
        sequence_length=50,
        num_sequences=10,
        trend_slope=0.1,
        noise_std=0.05,
        seed=42
    )
    
    sequences = list(dataset)
    
    assert len(sequences) == 10
    for seq in sequences:
        assert seq.shape == (50, 1)
        assert seq.dtype == torch.float32


def test_linear_trend_reproducibility():
    """Test that the dataset generates reproducible results with same seed."""
    dataset1 = LinearTrendDataset(
        sequence_length=20,
        num_sequences=5,
        seed=123
    )
    
    dataset2 = LinearTrendDataset(
        sequence_length=20,
        num_sequences=5,
        seed=123
    )
    
    sequences1 = list(dataset1)
    sequences2 = list(dataset2)
    
    for s1, s2 in zip(sequences1, sequences2):
        assert torch.allclose(s1, s2)


def test_linear_trend_parameters():
    """Test that trend slope and intercept affect the output correctly."""
    # Test with positive slope
    dataset_pos = LinearTrendDataset(
        sequence_length=10,
        num_sequences=1,
        trend_slope=1.0,
        noise_std=0.0,  # No noise for clear trend
        intercept=0.0,
        seed=42
    )
    
    seq_pos = list(dataset_pos)[0].squeeze()
    
    # Check that values generally increase (allowing for some noise)
    assert seq_pos[-1] > seq_pos[0]
    
    # Test with negative slope
    dataset_neg = LinearTrendDataset(
        sequence_length=10,
        num_sequences=1,
        trend_slope=-1.0,
        noise_std=0.0,
        intercept=10.0,
        seed=42
    )
    
    seq_neg = list(dataset_neg)[0].squeeze()
    
    # Check that values generally decrease
    assert seq_neg[-1] < seq_neg[0]


def test_linear_trend_with_builder():
    """Test LinearTrendDataset works with the Builder pattern."""
    from preprocessing.transform.dataset_builder import Builder
    
    dataset = LinearTrendDataset(
        sequence_length=10,
        num_sequences=5,
        seed=42
    )
    
    # Apply transformations using Builder
    processed = (
        Builder(dataset)
        .map(lambda x: x * 2.0)  # Scale by 2
        .batch(2)
        .build()
    )
    
    batches = list(processed)
    
    # Should have 2 full batches + 1 partial batch
    assert len(batches) == 3
    assert len(batches[0]) == 2  # First batch
    assert len(batches[1]) == 2  # Second batch
    assert len(batches[2]) == 1  # Last batch
    
    # Check that scaling was applied
    for batch in batches:
        for item in batch:
            assert item.shape == (10, 1)


def test_zero_noise():
    """Test dataset with zero noise produces exact linear trend."""
    dataset = LinearTrendDataset(
        sequence_length=5,
        num_sequences=1,
        trend_slope=2.0,
        noise_std=0.0,
        intercept=1.0,
        seed=42
    )
    
    seq = list(dataset)[0].squeeze()
    expected = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    
    assert torch.allclose(seq, expected)
