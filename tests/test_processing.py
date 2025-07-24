"""
Tests for stateless preprocessing utilities in processing.py

This module tests all transformation classes and the pipeline functionality
to ensure proper behavior for foundation time series models.
"""

import pytest
import torch
import numpy as np
from typing import Dict, List

from binconvfm.utils.processing import (
    BaseTransform,
    StandardScaler,
    TemporalScaler,
    BinaryQuantizer,
    TransformPipeline
)


class TestBaseTransform:
    """Test the base transformation class."""
    
    def test_base_transform_initialization(self):
        """Test that BaseTransform initializes correctly."""
        transform = BaseTransform()
        assert transform.input_dims is None
        assert transform.inverse_input_dims is None
    
    def test_shape_validation_error(self):
        """Test that shape validation raises appropriate errors."""
        transform = BaseTransform()
        transform.input_dims = 3
        
        # Create 2D tensor when 3D expected
        data = torch.randn(10, 5)
        
        with pytest.raises(ValueError, match="BaseTransform.transform"):
            transform._validate_input_shape(data, 3, "transform")
    
    def test_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        transform = BaseTransform()
        data = torch.randn(2, 10, 3)
        assert transform.fit(data) is transform
    
    def test_not_implemented_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        transform = BaseTransform()
        data = torch.randn(2, 10, 3)
        
        with pytest.raises(NotImplementedError):
            transform.transform(data)
        
        with pytest.raises(NotImplementedError):
            transform.inverse_transform(data, {})


class TestStandardScaler:
    """Test the StandardScaler transformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scaler = StandardScaler()
        self.batch_size = 4
        self.context_length = 20
        self.n_features = 3
        
        # Create test data with known statistics
        torch.manual_seed(42)
        self.data = torch.randn(self.batch_size, self.context_length, self.n_features)
        # Add some structure to make statistics more predictable
        self.data[:, :, 0] += 5.0  # Feature 0 has mean around 5
        self.data[:, :, 1] *= 2.0  # Feature 1 has larger variance
    
    def test_initialization(self):
        """Test that StandardScaler initializes correctly."""
        assert self.scaler.input_dims == 3
        assert self.scaler.inverse_input_dims == 3
    
    def test_shape_validation(self):
        """Test input shape validation."""
        # Test correct shape
        transformed, params = self.scaler.transform(self.data)
        assert transformed.shape == self.data.shape
        
        # Test incorrect shape
        wrong_shape_data = torch.randn(10, 5)  # 2D instead of 3D
        with pytest.raises(ValueError, match="StandardScaler.transform"):
            self.scaler.transform(wrong_shape_data)
    
    def test_transform_statistics(self):
        """Test that transform produces correct standardization."""
        transformed, params = self.scaler.transform(self.data)
        
        # Check that parameters have correct shapes
        assert params['mean'].shape == (self.batch_size, 1, self.n_features)
        assert params['std'].shape == (self.batch_size, 1, self.n_features)
        
        # Check that transformed data has approximately zero mean and unit std per sample
        for i in range(self.batch_size):
            sample_transformed = transformed[i]  # (context_length, n_features)
            
            # Mean should be close to 0
            sample_mean = sample_transformed.mean(dim=0)
            assert torch.allclose(sample_mean, torch.zeros_like(sample_mean), atol=1e-6)
            
            # Std should be close to 1
            sample_std = sample_transformed.std(dim=0, unbiased=False)
            assert torch.allclose(sample_std, torch.ones_like(sample_std), atol=1e-6)
    
    def test_inverse_transform(self):
        """Test that inverse transform recovers original data."""
        transformed, params = self.scaler.transform(self.data)
        reconstructed = self.scaler.inverse_transform(transformed, params)
        
        # Should recover original data
        assert torch.allclose(reconstructed, self.data, atol=1e-6)
    
    def test_device_handling(self):
        """Test that transformation works across different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Move data to GPU
        data_gpu = self.data.cuda()
        transformed, params = self.scaler.transform(data_gpu)
        
        # Parameters should be on GPU
        assert params['mean'].device == data_gpu.device
        assert params['std'].device == data_gpu.device
        
        # Move transformed data to CPU but keep params on GPU
        transformed_cpu = transformed.cpu()
        reconstructed = self.scaler.inverse_transform(transformed_cpu, params)
        
        # Should work and result should be on CPU
        assert reconstructed.device.type == 'cpu'
    
    def test_epsilon_handling(self):
        """Test that zero standard deviation is handled properly."""
        # Create data with zero variance in one feature
        constant_data = torch.ones(2, 10, 3)
        constant_data[:, :, 0] = 5.0  # Constant value
        
        transformed, params = self.scaler.transform(constant_data)
        
        # Should not contain NaN or inf
        assert torch.isfinite(transformed).all()
        assert torch.isfinite(params['std']).all()
        
        # Std should be greater than original zero std (due to added epsilon)
        eps = torch.finfo(constant_data.dtype).eps
        assert (params['std'][:, :, 0] >= eps).all()


class TestTemporalScaler:
    """Test the TemporalScaler transformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scaler = TemporalScaler()
        self.batch_size = 3
        self.context_length = 15
        self.n_features = 2
        
        # Create test data
        torch.manual_seed(42)
        self.data = torch.randn(self.batch_size, self.context_length, self.n_features) + 2.0
    
    def test_initialization(self):
        """Test that TemporalScaler initializes correctly."""
        assert self.scaler.input_dims == 3
        assert self.scaler.inverse_input_dims == 3
    
    def test_transform_scaling(self):
        """Test that temporal scaling works correctly."""
        transformed, params = self.scaler.transform(self.data)
        
        # Check parameter shape
        assert params['mean'].shape == (self.batch_size, 1, self.n_features)
        
        # Check that each sample is scaled by its per-feature mean (same as StandardScaler)
        for i in range(self.batch_size):
            sample = self.data[i]  # (context_length, n_features)
            sample_mean = sample.mean(dim=0, keepdim=True)  # (1, n_features)
            sample_mean = torch.abs(sample_mean) + torch.finfo(sample.dtype).eps
            expected_transformed = sample / sample_mean
            
            assert torch.allclose(transformed[i], expected_transformed, atol=1e-5)
    
    def test_inverse_transform(self):
        """Test that inverse transform recovers original data."""
        transformed, params = self.scaler.transform(self.data)
        reconstructed = self.scaler.inverse_transform(transformed, params)
        
        assert torch.allclose(reconstructed, self.data, atol=1e-6)
    
    def test_zero_mean_handling(self):
        """Test handling of data with zero per-feature mean."""
        # Create data with zero mean per feature
        zero_mean_data = torch.randn(2, 10, 2)
        # Make each feature have zero mean across time dimension
        zero_mean_data = zero_mean_data - zero_mean_data.mean(dim=1, keepdim=True)
        
        transformed, params = self.scaler.transform(zero_mean_data)
        
        # Should not contain NaN or inf
        assert torch.isfinite(transformed).all()
        assert torch.isfinite(params['mean']).all()
        
        # Mean should be epsilon, not zero (due to abs() + eps)
        eps = torch.finfo(zero_mean_data.dtype).eps
        assert (torch.abs(params['mean']) >= eps).all()


class TestBinaryQuantizer:
    """Test the BinaryQuantizer transformation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.num_bins = 10
        self.min_val = -2.0
        self.max_val = 2.0
        self.quantizer = BinaryQuantizer(
            num_bins=self.num_bins,
            min_val=self.min_val,
            max_val=self.max_val
        )
        
        self.batch_size = 2
        self.context_length = 8
        self.n_features = 2
        
        # Create test data within quantization range
        torch.manual_seed(42)
        self.data = torch.empty(self.batch_size, self.context_length, self.n_features).uniform_(
            self.min_val,
            self.max_val
        )
    
    def test_initialization(self):
        """Test that BinaryQuantizer initializes correctly."""
        assert self.quantizer.input_dims == 3
        assert self.quantizer.inverse_input_dims == 4
        assert self.quantizer.num_bins == self.num_bins
        assert self.quantizer.min_val == self.min_val
        assert self.quantizer.max_val == self.max_val
        
        # Check bin parameters
        assert self.quantizer.bin_edges.shape == (self.num_bins + 1,)
        assert self.quantizer.bin_values.shape == (self.num_bins,)
    
    def test_transform_shape(self):
        """Test that transform produces correct output shape."""
        transformed, params = self.quantizer.transform(self.data)
        
        expected_shape = (self.batch_size, self.context_length, self.n_features, self.num_bins)
        assert transformed.shape == expected_shape
        
        # Params should be empty for quantizer
        assert params == {}
    
    def test_binary_property(self):
        """Test that output is binary (0s and 1s only)."""
        transformed, _ = self.quantizer.transform(self.data)
        
        # All values should be 0 or 1
        assert ((transformed == 0) | (transformed == 1)).all()
    
    def test_monotonic_property(self):
        """Test that binary vectors are monotonic (1s followed by 0s)."""
        # Create simple test case
        simple_data = torch.tensor([[[0.0], [1.0], [-1.0]]])  # (1, 3, 1)
        transformed, _ = self.quantizer.transform(simple_data)
        
        # For each position, check monotonicity
        for b in range(transformed.shape[0]):
            for t in range(transformed.shape[1]):
                for f in range(transformed.shape[2]):
                    binary_vec = transformed[b, t, f]
                    
                    # Find first zero
                    zeros = (binary_vec == 0).nonzero(as_tuple=False)
                    if len(zeros) > 0:
                        first_zero = zeros[0].item()
                        # All positions after first zero should be zero
                        assert (binary_vec[first_zero:] == 0).all()
    
    def test_inverse_transform_shape(self):
        """Test that inverse transform produces correct shape."""
        transformed, params = self.quantizer.transform(self.data)
        reconstructed = self.quantizer.inverse_transform(transformed, params)
        
        assert reconstructed.shape == self.data.shape
    
    def test_inverse_transform_range(self):
        """Test that inverse transform stays within expected range."""
        transformed, params = self.quantizer.transform(self.data)
        reconstructed = self.quantizer.inverse_transform(transformed, params)
        
        # Reconstructed values should be within [min_val, max_val]
        assert (reconstructed >= self.min_val).all()
        assert (reconstructed <= self.max_val).all()
    
    def test_shape_validation(self):
        """Test input shape validation for both methods."""
        # Test transform with wrong shape
        wrong_shape = torch.randn(10, 5)  # 2D instead of 3D
        with pytest.raises(ValueError, match="BinaryQuantizer.transform"):
            self.quantizer.transform(wrong_shape)
        
        # Test inverse_transform with wrong shape (3D instead of 4D)
        wrong_shape_3d = torch.randn(2, 8, 2)  # 3D instead of 4D
        with pytest.raises(ValueError, match="BinaryQuantizer.inverse_transform"):
            self.quantizer.inverse_transform(wrong_shape_3d, {})


class TestTransformPipeline:
    """Test the TransformPipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scaler = StandardScaler()
        self.quantizer = BinaryQuantizer(num_bins=5, min_val=-3.0, max_val=3.0)
        
        self.pipeline = TransformPipeline([
            ('scaler', self.scaler),
            ('quantizer', self.quantizer)
        ])
        
        self.batch_size = 2
        self.context_length = 10
        self.n_features = 2
        
        torch.manual_seed(42)
        self.data = torch.randn(self.batch_size, self.context_length, self.n_features)
    
    def test_initialization(self):
        """Test that pipeline initializes correctly."""
        assert len(self.pipeline.steps) == 2
        assert self.pipeline.step_names == ['scaler', 'quantizer']
        assert len(self.pipeline.transformers) == 2
    
    def test_transform_sequence(self):
        """Test that pipeline applies transformations in sequence."""
        final_output, all_params = self.pipeline.transform(self.data)
        
        # Should have parameters from each step
        assert len(all_params) == 2
        
        # First step parameters (scaler)
        scaler_params = all_params[0]
        assert 'mean' in scaler_params
        assert 'std' in scaler_params
        
        # Second step parameters (quantizer - empty)
        quantizer_params = all_params[1]
        assert quantizer_params == {}
        
        # Final output should be 4D (from quantizer)
        expected_shape = (self.batch_size, self.context_length, self.n_features, 5)
        assert final_output.shape == expected_shape
        
        # Should be binary
        assert ((final_output == 0) | (final_output == 1)).all()
    
    def test_inverse_transform_sequence(self):
        """Test that pipeline applies inverse transforms in reverse order."""
        transformed, all_params = self.pipeline.transform(self.data)
        reconstructed = self.pipeline.inverse_transform(transformed, all_params)
        
        # Should recover approximately original data
        # (some loss due to quantization)
        assert reconstructed.shape == self.data.shape
        
        # Check that reconstruction is reasonable (not exact due to quantization)
        # Use a more lenient test for quantization error
        absolute_error = torch.abs(reconstructed - self.data)
        assert absolute_error.mean() < 2.0  # Reasonable approximation given coarse quantization
    
    def test_fit_method(self):
        """Test that fit method works for compatibility."""
        result = self.pipeline.fit(self.data)
        assert result is self.pipeline
    
    def test_empty_pipeline(self):
        """Test edge case of empty pipeline."""
        empty_pipeline = TransformPipeline([])
        
        transformed, params = empty_pipeline.transform(self.data)
        
        # Should return original data and empty params
        assert torch.equal(transformed, self.data)
        assert params == []
        
        # Inverse should also return original data
        reconstructed = empty_pipeline.inverse_transform(self.data, [])
        assert torch.equal(reconstructed, self.data)
    
    def test_single_transform_pipeline(self):
        """Test pipeline with single transformation."""
        single_pipeline = TransformPipeline([('scaler', StandardScaler())])
        
        transformed, params = single_pipeline.transform(self.data)
        reconstructed = single_pipeline.inverse_transform(transformed, params)
        
        # Should be exact reconstruction for StandardScaler
        assert torch.allclose(reconstructed, self.data, atol=1e-6)


class TestIntegration:
    """Integration tests for the complete preprocessing system."""
    
    def test_different_data_types(self):
        """Test that transformations work with different data types."""
        data_types = [torch.float32, torch.float64]
        
        for dtype in data_types:
            data = torch.randn(2, 10, 3, dtype=dtype)
            
            scaler = StandardScaler()
            transformed, params = scaler.transform(data)
            reconstructed = scaler.inverse_transform(transformed, params)
            
            assert transformed.dtype == dtype
            assert reconstructed.dtype == dtype
            assert torch.allclose(reconstructed, data, atol=1e-6)
    
    def test_batch_independence(self):
        """Test that samples in batch are processed independently."""
        # Create batch where samples have very different statistics
        batch_size = 3
        context_length = 20
        n_features = 2
        
        data = torch.zeros(batch_size, context_length, n_features)
        data[0] = torch.randn(context_length, n_features) * 0.1  # Small variance
        data[1] = torch.randn(context_length, n_features) * 10.0 + 100.0  # Large mean, large variance
        data[2] = torch.randn(context_length, n_features) * 2.0 - 50.0  # Negative mean
        
        scaler = StandardScaler()
        transformed, params = scaler.transform(data)
        
        # Each sample should be standardized independently
        for i in range(batch_size):
            sample_mean = transformed[i].mean(dim=0)
            sample_std = transformed[i].std(dim=0, unbiased=False)
            
            assert torch.allclose(sample_mean, torch.zeros_like(sample_mean), atol=1e-5)
            assert torch.allclose(sample_std, torch.ones_like(sample_std), atol=1e-5)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Single sample batch
        single_batch = torch.randn(1, 10, 3)
        scaler = StandardScaler()
        transformed, params = scaler.transform(single_batch)
        reconstructed = scaler.inverse_transform(transformed, params)
        assert torch.allclose(reconstructed, single_batch, atol=1e-6)
        
        # Single feature
        single_feature = torch.randn(2, 10, 1)
        transformed, params = scaler.transform(single_feature)
        reconstructed = scaler.inverse_transform(transformed, params)
        assert torch.allclose(reconstructed, single_feature, atol=1e-6)
        
        # Minimum context length
        min_context = torch.randn(2, 1, 3)
        transformed, params = scaler.transform(min_context)
        reconstructed = scaler.inverse_transform(transformed, params)
        assert torch.allclose(reconstructed, min_context, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])