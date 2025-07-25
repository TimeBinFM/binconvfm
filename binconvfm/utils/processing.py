"""
Stateless preprocessing utilities for foundation time series models.

This module provides stateless preprocessing transformations that compute parameters per sample
and apply transformations independently, making them ideal for foundation models where
each time series should be processed without cross-sample information leakage.

The core design:
1. Each transformation class has transform() and inverse_transform() methods
2. transform() returns (transformed_data, params) tuple
3. inverse_transform() takes (data, params) and returns original data
4. Pipeline class chains multiple transformations while maintaining statelessness
5. fit() methods are compatibility stubs that return self
6. Each transformer defines its expected input dimensions and shape descriptions in __init__()

This approach enables efficient batch processing while maintaining the per-sample 
independence required for foundation models.

Example usage:
    # Create individual transformers
    scaler = StandardScaler()
    quantizer = BinaryQuantizer(num_bins=1000, min_val=-5.0, max_val=5.0)
    
    # Create pipeline
    pipeline = TransformPipeline([
        ('scaler', scaler),
        ('quantizer', quantizer)
    ])
    
    # Use pipeline
    transformed_data, params = pipeline.transform(data)
    reconstructed_data = pipeline.inverse_transform(transformed_data, params)
"""

import torch
import logging
from typing import Dict, Tuple, Union, List, Any

# Set up logger
logger = logging.getLogger(__name__)


# =============================================================================
# Base Transformation Class
# =============================================================================

class BaseTransform:
    """Base class for stateless transformations."""
    
    def __init__(self):
        """Initialize base transform. Subclasses should set input dimensions and shape descriptions."""
        self.input_dims = None
        self.inverse_input_dims = None
    
    def _get_shape_description(self, dims: int) -> str:
        """
        Get human-readable shape description for given dimensions.
        Subclasses should override this method to provide specific descriptions.
        """
        return f"{dims}D tensor"
    
    def _validate_input_shape(self, data: torch.Tensor, expected_dims: int, operation: str = "transform") -> None:
        """
        Validate input tensor shape.
        
        Args:
            data: Input tensor to validate
            expected_dims: Expected number of dimensions
            operation: Name of operation for error message
            
        Raises:
            ValueError: If tensor doesn't have expected dimensions
        """
        if data.dim() != expected_dims:
            expected_shape = self._get_shape_description(expected_dims)
            raise ValueError(f"{self.__class__.__name__}.{operation}() expected {expected_dims}D tensor "
                           f"with shape {expected_shape}, but got {data.dim()}D tensor with shape {data.shape}")
    
    def fit(self, data: torch.Tensor) -> 'BaseTransform':
        """Compatibility method - returns self since transformations are stateless."""
        return self
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transform data and return transformed data with parameters.
        
        Args:
            data: Input tensor
            params: Optional pre-computed parameters. If provided, uses these instead of computing new ones.
            
        Returns:
            Tuple of (transformed_data, params)
        """
        if self.input_dims is not None:
            self._validate_input_shape(data, self.input_dims, "transform")
        raise NotImplementedError
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inverse transform data using provided parameters.
        
        Args:
            data: Transformed tensor
            params: Parameters from transform() method
            
        Returns:
            Original scale tensor
        """
        if self.inverse_input_dims is not None:
            self._validate_input_shape(data, self.inverse_input_dims, "inverse_transform")
        raise NotImplementedError


# =============================================================================
# Individual Transformation Classes
# =============================================================================

class StandardScaler(BaseTransform):
    """
    Stateless standard scaler for foundation models.
    
    Computes standardization parameters per sample and applies them independently.
    """
    
    def __init__(self):
        """Initialize the standard scaler."""
        super().__init__()
        self.input_dims = 3  # (batch_size, context_length, n_features)
        self.inverse_input_dims = 3  # (batch_size, context_length, n_features)
        logger.debug('StandardScaler initialized for foundation model use')
    
    def _get_shape_description(self, dims: int) -> str:
        """Get shape description specific to StandardScaler."""
        if dims == 3:
            return "(batch_size, context_length, n_features)"
        else:
            return f"{dims}D tensor"
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transform data using per-sample standardization.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            params: Optional pre-computed parameters. If provided, uses these instead of computing new ones.
                   Should contain 'mean' and 'std' keys.
            
        Returns:
            Tuple of (standardized_data, params)
        """
        self._validate_input_shape(data, self.input_dims, "transform")
        
        if params is not None:
            # Use provided parameters
            mean = params['mean'].to(data.device)
            std = params['std'].to(data.device)
        else:
            # Compute per-target statistics for each sample
            mean = data.mean(dim=1, keepdim=True)  # (batch_size, 1, n_features)
            std = data.std(dim=1, keepdim=True, unbiased=False)  # (batch_size, 1, n_features)
            
            # Prevent division by zero by adding appropriate epsilon
            eps = torch.finfo(data.dtype).eps
            std = std + eps
        
        # Apply transformation
        transformed = (data - mean) / std
        
        # Store parameters
        result_params = {'mean': mean, 'std': std}
        
        return transformed, result_params
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inverse transform standardized data.
        
        Args:
            data: Standardized tensor of shape (batch_size, context_length, n_features)
            params: Parameters from transform() method
            
        Returns:
            Original scale tensor
        """
        self._validate_input_shape(data, self.inverse_input_dims, "inverse_transform")
        
        mean = params['mean'].to(data.device)
        std = params['std'].to(data.device)
        
        # Apply inverse transformation
        return data * std + mean


class TemporalScaler(BaseTransform):
    """
    Stateless temporal scaler for foundation models.
    
    Scales data by dividing by the global mean, preserving relative temporal patterns.
    """
    
    def __init__(self):
        """Initialize the temporal scaler."""
        super().__init__()
        self.input_dims = 3  # (batch_size, context_length, n_features)
        self.inverse_input_dims = 3  # (batch_size, context_length, n_features)
        logger.debug('TemporalScaler initialized for foundation model use')
    
    def _get_shape_description(self, dims: int) -> str:
        """Get shape description specific to TemporalScaler."""
        if dims == 3:
            return "(batch_size, context_length, n_features)"
        else:
            return f"{dims}D tensor"
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transform data using per-sample temporal scaling.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            params: Optional pre-computed parameters. If provided, uses these instead of computing new ones.
                   Should contain 'mean' key.
            
        Returns:
            Tuple of (scaled_data, params)
        """
        self._validate_input_shape(data, self.input_dims, "transform")
        
        if params is not None:
            # Use provided parameters
            mean = params['mean'].to(data.device)
        else:
            # Compute mean for each sample and feature (same as StandardScaler)
            mean = data.mean(dim=1, keepdim=True)  # (batch_size, 1, n_features)
            
            # Take absolute value and add epsilon to prevent division by zero
            eps = torch.finfo(data.dtype).eps
            mean = torch.abs(mean) + eps
        
        # Apply transformation
        transformed = data / mean
        
        # Store parameters
        result_params = {'mean': mean}
        
        return transformed, result_params
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inverse transform scaled data.
        
        Args:
            data: Scaled tensor of shape (batch_size, context_length, n_features)
            params: Parameters from transform() method
            
        Returns:
            Original scale tensor
        """
        self._validate_input_shape(data, self.inverse_input_dims, "inverse_transform")
        
        mean = params['mean'].to(data.device)
        
        # Apply inverse transformation
        return data * mean


class BinaryQuantizer(BaseTransform):
    """
    Stateless binary quantizer for foundation models.
    
    Transforms continuous values into binary vectors using fixed bin thresholds.
    Bin parameters are stateful (class attributes) since they should be consistent.
    """
    
    def __init__(self, num_bins: int = 1000, min_val: float = -10.0, max_val: float = 10.0):
        """
        Initialize the binary quantizer.
        
        Args:
            num_bins: Number of quantization bins
            min_val: Minimum value for quantization range
            max_val: Maximum value for quantization range
        """
        super().__init__()
        self.input_dims = 3  # (batch_size, context_length, n_features)
        self.inverse_input_dims = 4  # (batch_size, context_length, n_features, num_bins)
        
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        
        # Compute quantization parameters (stateful - same for all samples)
        self.bin_edges = torch.linspace(min_val, max_val, num_bins + 1)
        self.bin_values = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        
        logger.debug(f'BinaryQuantizer initialized - num_bins: {num_bins}, '
                    f'min_val: {min_val}, max_val: {max_val}')
    
    def _get_shape_description(self, dims: int) -> str:
        """Get shape description specific to BinaryQuantizer."""
        if dims == 3:
            return "(batch_size, context_length, n_features)"
        elif dims == 4:
            return "(batch_size, context_length, n_features, num_bins)"
        else:
            return f"{dims}D tensor"
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transform data to binary quantized representation.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            params: Optional pre-computed parameters. For BinaryQuantizer, this is ignored 
                   since quantization parameters are stored in the object itself.
            
        Returns:
            Tuple of (binary_data, empty_params) where binary_data has shape 
            (batch_size, context_length, n_features, num_bins)
        """
        self._validate_input_shape(data, self.input_dims, "transform")
        
        # Note: params argument is ignored for BinaryQuantizer since bin parameters
        # are stateful and stored in the object itself
        
        # Move bin_edges to same device as data
        bin_edges = self.bin_edges.to(data.device)
        
        # Add dimension for broadcasting: (batch_size, context_length, n_features, 1)
        data_expanded = data.unsqueeze(-1)
        
        # Create thresholds for broadcasting: (num_bins,)
        bin_thresholds = bin_edges[1:]  # Skip the first edge
        
        # Broadcast comparison: (batch_size, context_length, n_features, num_bins)
        binary_vectors = (data_expanded >= bin_thresholds).float()
        
        # Return empty params dict for consistency (params are stored in object)
        result_params = {}
        
        return binary_vectors, result_params
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inverse transform binary quantized data back to continuous values.
        
        Args:
            data: Binary tensor of shape (batch_size, context_length, n_features, num_bins)
            params: Empty params dict (not used, kept for consistency)
            
        Returns:
            Reconstructed continuous tensor of shape (batch_size, context_length, n_features)
        """
        self._validate_input_shape(data, self.inverse_input_dims, "inverse_transform")
        
        # Use internal parameters (bin_values stored in object)
        bin_values = self.bin_values.to(data.device)
        
        # Find the last active bin (highest bin that's active)
        # Flip the binary vectors to find the first 1 from the right
        reversed_bin = torch.flip(data, dims=[-1])
        idx_first_one_reversed = torch.argmax(reversed_bin, dim=-1)
        idx_last_one = self.num_bins - 1 - idx_first_one_reversed
        
        # Get the corresponding bin values
        reconstructed = bin_values[idx_last_one]
        
        # Handle edge case: all zeros (no active bins)
        all_zero_mask = data.sum(dim=-1) == 0
        reconstructed = torch.where(all_zero_mask, bin_values[0], reconstructed)
        
        return reconstructed


# =============================================================================
# Pipeline Class
# =============================================================================

class TransformPipeline:
    """
    Stateless pipeline for chaining multiple transformations.
    
    Maintains parameter storage for each transformation step to enable
    proper inverse transformation while keeping the pipeline stateless.
    """
    
    def __init__(self, steps: List[Tuple[str, BaseTransform]]):
        """
        Initialize the pipeline.
        
        Args:
            steps: List of (name, transformer) tuples
        """
        self.steps = steps
        self.step_names = [name for name, _ in steps]
        self.transformers = [transformer for _, transformer in steps]
        
        logger.debug(f'TransformPipeline initialized with steps: {self.step_names}')
    
    def fit(self, data: torch.Tensor) -> 'TransformPipeline':
        """
        Fit all transformers (compatibility method - returns self since stateless).
        
        Args:
            data: Input tensor to fit on
            
        Returns:
            self for method chaining
        """
        for transformer in self.transformers:
            transformer.fit(data)
        return self
    
    def transform(self, data: torch.Tensor, all_params: List[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Apply all transformations in sequence.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            all_params: Optional pre-computed parameters for all steps. If provided, 
                       uses these instead of computing new ones.
            
        Returns:
            Tuple of (final_transformed_data, list_of_params_per_step)
        """
        current_data = data
        result_params = []
        
        for i, (name, transformer) in enumerate(self.steps):
            logger.debug(f'Applying transformation step {i+1}/{len(self.steps)}: {name}')
            
            # Use provided params if available, otherwise None (compute new ones)
            step_params = all_params[i] if all_params is not None else None
            
            current_data, params = transformer.transform(current_data, step_params)
            result_params.append(params)
        
        return current_data, result_params
    
    def inverse_transform(self, data: torch.Tensor, all_params: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Apply inverse transformations in reverse order.
        
        Args:
            data: Transformed tensor to inverse transform
            all_params: List of parameters from transform() method
            
        Returns:
            Original scale tensor
        """
        current_data = data
        
        # Apply inverse transformations in reverse order
        for i, ((name, transformer), params) in enumerate(zip(reversed(self.steps), reversed(all_params))):
            step_num = len(self.steps) - i
            logger.debug(f'Applying inverse transformation step {step_num}/{len(self.steps)}: {name}')
            current_data = transformer.inverse_transform(current_data, params)
        
        return current_data