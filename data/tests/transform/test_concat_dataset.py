import torch
import pytest

from preprocessing.common import TensorIterableDataset
from preprocessing.transform.concat_dataset import ConcatDataset

# Dummy datasets that emit single-item tensors
class DummyDatasetA(TensorIterableDataset):
    def __iter__(self):
        for i in range(3):
            yield torch.tensor([i], dtype=torch.float32)

class DummyDatasetB(TensorIterableDataset):
    def __iter__(self):
        for i in range(3, 6):
            yield torch.tensor([i], dtype=torch.float32)

class DummyDatasetC(TensorIterableDataset):
    def __iter__(self):
        for i in range(6, 9):
            yield torch.tensor([i], dtype=torch.float32)

class EmptyDataset(TensorIterableDataset):
    def __iter__(self):
        return iter([])

def test_concat_dataset_basic():
    """Test basic concatenation of two datasets"""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    concat_ds = ConcatDataset([ds1, ds2])
    
    expected = [torch.tensor([i], dtype=torch.float32) for i in range(6)]
    output = list(concat_ds)
    
    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"

def test_concat_dataset_single():
    """Test concatenation with a single dataset"""
    ds1 = DummyDatasetA()
    
    concat_ds = ConcatDataset([ds1])
    
    expected = [torch.tensor([i], dtype=torch.float32) for i in range(3)]
    output = list(concat_ds)
    
    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"

def test_concat_dataset_multiple():
    """Test concatenation of three datasets"""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    ds3 = DummyDatasetC()
    
    concat_ds = ConcatDataset([ds1, ds2, ds3])
    
    expected = [torch.tensor([i], dtype=torch.float32) for i in range(9)]
    output = list(concat_ds)
    
    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"

def test_concat_dataset_with_empty():
    """Test concatenation including empty datasets"""
    ds1 = DummyDatasetA()
    empty_ds = EmptyDataset()
    ds2 = DummyDatasetB()
    
    concat_ds = ConcatDataset([ds1, empty_ds, ds2])
    
    expected = [torch.tensor([i], dtype=torch.float32) for i in range(6)]
    output = list(concat_ds)
    
    assert len(output) == len(expected)
    for out, exp in zip(output, expected):
        assert torch.equal(out, exp), f"Expected {exp}, got {out}"

def test_concat_dataset_all_empty():
    """Test concatenation of only empty datasets"""
    empty_ds1 = EmptyDataset()
    empty_ds2 = EmptyDataset()
    
    concat_ds = ConcatDataset([empty_ds1, empty_ds2])
    
    output = list(concat_ds)
    assert len(output) == 0

def test_concat_dataset_empty_list():
    """Test concatenation with empty list of datasets"""
    concat_ds = ConcatDataset([])
    
    output = list(concat_ds)
    assert len(output) == 0

def test_concat_dataset_iteration_order():
    """Test that items are yielded in the correct order"""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    concat_ds = ConcatDataset([ds1, ds2])
    
    # Test that we get items from ds1 first, then ds2
    output = list(concat_ds)
    
    # First 3 items should be from ds1 (0, 1, 2)
    for i in range(3):
        assert torch.equal(output[i], torch.tensor([i], dtype=torch.float32))
    
    # Next 3 items should be from ds2 (3, 4, 5)
    for i in range(3):
        assert torch.equal(output[i + 3], torch.tensor([i + 3], dtype=torch.float32))

def test_concat_dataset_multiple_iterations():
    """Test that the dataset can be iterated multiple times"""
    ds1 = DummyDatasetA()
    ds2 = DummyDatasetB()
    
    concat_ds = ConcatDataset([ds1, ds2])
    
    # First iteration
    output1 = list(concat_ds)
    expected = [torch.tensor([i], dtype=torch.float32) for i in range(6)]
    
    assert len(output1) == len(expected)
    for out, exp in zip(output1, expected):
        assert torch.equal(out, exp)
    
    # Second iteration
    output2 = list(concat_ds)
    
    assert len(output2) == len(expected)
    for out, exp in zip(output2, expected):
        assert torch.equal(out, exp)

def test_concat_dataset_tensor_shapes():
    """Test concatenation with tensors of different shapes"""
    class ShapeDatasetA(TensorIterableDataset):
        def __iter__(self):
            yield torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
            yield torch.tensor([[5, 6]], dtype=torch.float32)
    
    class ShapeDatasetB(TensorIterableDataset):
        def __iter__(self):
            yield torch.tensor([7, 8, 9], dtype=torch.float32)
            yield torch.tensor([10], dtype=torch.float32)
    
    ds1 = ShapeDatasetA()
    ds2 = ShapeDatasetB()
    
    concat_ds = ConcatDataset([ds1, ds2])
    output = list(concat_ds)
    
    expected_shapes = [(2, 2), (1, 2), (3,), (1,)]
    assert len(output) == len(expected_shapes)
    
    for tensor, expected_shape in zip(output, expected_shapes):
        assert tensor.shape == expected_shape 
