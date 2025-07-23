"""
Stateless preprocessing utilities for foundation time series models.

This module provides stateless preprocessing functions that compute parameters per sample
and apply transformations independently, making them ideal for foundation models where
each time series should be processed without cross-sample information leakage.

The core philosophy is:
1. compute_params_batch() - Compute preprocessing parameters per sample in a batch
2. transform_batch() - Apply transformations using per-sample parameters  
3. inverse_transform_batch() - Reverse transformations using per-sample parameters

This approach enables efficient batch processing while maintaining the per-sample 
independence required for foundation models.
"""

import torch
import logging
from typing import Dict, Tuple, Union, Literal, Any
import warnings

# Set up logger
logger = logging.getLogger(__name__)


# =============================================================================
# Core Stateless Functions for Foundation Models
# =============================================================================

def compute_standard_params_batch(batch_data: torch.Tensor, var_specific: bool = False) -> Dict[str, torch.Tensor]:
    """
    Compute standard scaling parameters per sample in a batch.
    
    For foundation models, each sample's parameters are computed independently
    without any cross-sample information leakage.
    
    Args:
        batch_data: Input tensor of shape (batch_size, seq_len, features)
        var_specific: If True, compute per-feature statistics. If False, global statistics.
        
    Returns:
        Dict containing 'mean' and 'std' tensors with appropriate shapes:
        - If var_specific=False: shape (batch_size, 1, 1) 
        - If var_specific=True: shape (batch_size, 1, features)
    """
    batch_size, seq_len, features = batch_data.shape
    
    if var_specific:
        # Compute per-feature statistics for each sample
        mean = batch_data.mean(dim=1, keepdim=True)  # (batch_size, 1, features)
        std = batch_data.std(dim=1, keepdim=True, unbiased=False)  # (batch_size, 1, features)
    else:
        # Compute global statistics for each sample
        mean = batch_data.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)  # (batch_size, 1, 1)
        std = batch_data.std(dim=(1, 2), keepdim=True, unbiased=False).unsqueeze(-1)  # (batch_size, 1, 1)
    
    # Prevent division by zero
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    return {'mean': mean, 'std': std}


def compute_temporal_params_batch(batch_data: torch.Tensor, time_first: bool = True) -> Dict[str, torch.Tensor]:
    """
    Compute temporal scaling parameters per sample in a batch.
    
    Temporal scaling normalizes by the mean of each sample, preserving relative patterns.
    
    Args:
        batch_data: Input tensor of shape (batch_size, seq_len, features)
        time_first: If True, time dimension is first after batch. Currently ignored.
        
    Returns:
        Dict containing 'mean' tensor of shape (batch_size, 1, 1)
    """
    # Compute global mean for each sample
    mean = batch_data.mean(dim=(1, 2), keepdim=True).unsqueeze(-1)  # (batch_size, 1, 1)
    
    # Prevent division by zero
    mean = torch.where(mean == 0, torch.ones_like(mean), mean)
    
    return {'mean': mean}


def compute_binary_quantization_params(num_bins: int, min_val: float = -10.0, max_val: float = 10.0) -> Dict[str, torch.Tensor]:
    """
    Compute binary quantization parameters (stateless - same for all samples).
    
    Args:
        num_bins: Number of quantization bins
        min_val: Minimum value for quantization range
        max_val: Maximum value for quantization range
        
    Returns:
        Dict containing 'bin_edges' and 'bin_values' tensors
    """
    bin_edges = torch.linspace(min_val, max_val, num_bins + 1)
    bin_values = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return {'bin_edges': bin_edges, 'bin_values': bin_values, 'num_bins': num_bins}


def transform_standard_batch(batch_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply standard scaling transformation to batch using per-sample parameters.
    
    Args:
        batch_data: Input tensor of shape (batch_size, seq_len, features)
        params: Parameters dict with 'mean' and 'std' tensors
        
    Returns:
        Standardized tensor of same shape as input
    """
    mean = params['mean'].to(batch_data.device)
    std = params['std'].to(batch_data.device)
    
    return (batch_data - mean) / std


def transform_temporal_batch(batch_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply temporal scaling transformation to batch using per-sample parameters.
    
    Args:
        batch_data: Input tensor of shape (batch_size, seq_len, features)  
        params: Parameters dict with 'mean' tensor
        
    Returns:
        Scaled tensor of same shape as input
    """
    mean = params['mean'].to(batch_data.device)
    
    return batch_data / mean


def transform_binary_quantization_batch(batch_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply binary quantization transformation to batch.
    
    Creates binary vectors where element i is 1 if the input value is greater than
    or equal to the i-th bin threshold, 0 otherwise.
    
    Args:
        batch_data: Input tensor of shape (batch_size, seq_len, features)
        params: Parameters dict with 'bin_edges' tensor
        
    Returns:
        Binary tensor of shape (batch_size, seq_len, features, num_bins)
    """
    bin_edges = params['bin_edges'].to(batch_data.device)
    
    # Add dimension for broadcasting: (batch_size, seq_len, features, 1)
    data_expanded = batch_data.unsqueeze(-1)
    
    # Create thresholds for broadcasting: (num_bins,)
    bin_thresholds = bin_edges[1:]  # Skip the first edge
    
    # Broadcast comparison: (batch_size, seq_len, features, num_bins)
    binary_vectors = (data_expanded >= bin_thresholds).float()
    
    return binary_vectors


def inverse_transform_standard_batch(batch_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply inverse standard scaling transformation to batch.
    
    Args:
        batch_data: Standardized tensor of shape (batch_size, seq_len, features)
        params: Parameters dict with 'mean' and 'std' tensors
        
    Returns:
        Original scale tensor of same shape as input
    """
    mean = params['mean'].to(batch_data.device)
    std = params['std'].to(batch_data.device)
    
    return batch_data * std + mean


def inverse_transform_temporal_batch(batch_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply inverse temporal scaling transformation to batch.
    
    Args:
        batch_data: Scaled tensor of shape (batch_size, seq_len, features)
        params: Parameters dict with 'mean' tensor
        
    Returns:
        Original scale tensor of same shape as input
    """
    mean = params['mean'].to(batch_data.device)
    
    return batch_data * mean


def inverse_transform_binary_quantization_batch(batch_data: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Apply inverse binary quantization transformation to batch.
    
    For each binary vector, finds the highest active bin and returns the 
    corresponding bin center value.
    
    Args:
        batch_data: Binary tensor of shape (batch_size, seq_len, features, num_bins)
        params: Parameters dict with 'bin_values' tensor
        
    Returns:
        Reconstructed tensor of shape (batch_size, seq_len, features)
    """
    bin_values = params['bin_values'].to(batch_data.device)
    
    # Find the last active bin (highest bin that's active)
    # Flip the binary vectors to find the first 1 from the right
    reversed_bin = torch.flip(batch_data, dims=[-1])
    idx_first_one_reversed = torch.argmax(reversed_bin, dim=-1)
    idx_last_one = params['num_bins'] - 1 - idx_first_one_reversed
    
    # Get the corresponding bin values
    reconstructed = bin_values[idx_last_one]
    
    # Handle edge case: all zeros (no active bins)
    all_zero_mask = batch_data.sum(dim=-1) == 0
    reconstructed = torch.where(all_zero_mask, bin_values[0], reconstructed)
    
    return reconstructed


# =============================================================================
# Foundation Model Compatible Classes
# =============================================================================

class StandardScaler:
    """
    Stateless standard scaler for foundation models.
    
    Computes standardization parameters per sample and applies them independently.
    Compatible with the original BinConv interface.
    """
    
    def __init__(self, var_specific: bool = False):
        """
        Initialize the standard scaler.
        
        Args:
            var_specific: If True, compute per-feature statistics. If False, global statistics.
        """
        self.var_specific = var_specific
        logger.debug(f'StandardScaler initialized with var_specific={var_specific}')
    
    def fit(self, data: torch.Tensor) -> 'StandardScaler':
        """Compatibility method - returns self since we're stateless."""
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data using per-sample standardization.
        
        Args:
            data: Input tensor of shape (batch_size, seq_len, features) or (seq_len, features)
            
        Returns:
            Standardized tensor of same shape
        """
        # Handle single sample case
        if data.dim() == 2:  # (seq_len, features)
            data = data.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
        
        # Compute parameters and transform
        params = compute_standard_params_batch(data, self.var_specific)
        transformed = transform_standard_batch(data, params)
        
        if single_sample:
            transformed = transformed.squeeze(0)  # Remove batch dimension
        
        return transformed
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform data (requires refitting on each call for stateless operation).
        
        Args:
            data: Standardized tensor
            
        Returns:
            Original scale tensor
        """
        warnings.warn("inverse_transform for StandardScaler requires the original data statistics. "
                     "For foundation models, store parameters explicitly.", UserWarning)
        # For stateless operation, we cannot perform inverse transform without stored parameters
        return data


class TemporalScaler:
    """
    Stateless temporal scaler for foundation models.
    
    Scales data by dividing by the mean, preserving relative temporal patterns.
    """
    
    def __init__(self, time_first: bool = True):
        """
        Initialize the temporal scaler.
        
        Args:
            time_first: If True, time dimension comes first after batch. Currently ignored.
        """
        self.time_first = time_first
        logger.debug(f'TemporalScaler initialized with time_first={time_first}')
    
    def fit(self, data: torch.Tensor) -> 'TemporalScaler':
        """Compatibility method - returns self since we're stateless."""
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data using per-sample temporal scaling.
        
        Args:
            data: Input tensor of shape (batch_size, seq_len, features) or (seq_len, features)
            
        Returns:
            Scaled tensor of same shape
        """
        # Handle single sample case
        if data.dim() == 2:  # (seq_len, features)
            data = data.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
        
        # Compute parameters and transform
        params = compute_temporal_params_batch(data, self.time_first)
        transformed = transform_temporal_batch(data, params)
        
        if single_sample:
            transformed = transformed.squeeze(0)  # Remove batch dimension
        
        return transformed
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform data (requires refitting on each call for stateless operation).
        
        Args:
            data: Scaled tensor
            
        Returns:
            Original scale tensor
        """
        warnings.warn("inverse_transform for TemporalScaler requires the original data statistics. "
                     "For foundation models, store parameters explicitly.", UserWarning)
        # For stateless operation, we cannot perform inverse transform without stored parameters
        return data


class BinaryQuantizer:
    """
    Stateless binary quantizer for foundation models.
    
    Transforms continuous values into binary vectors using fixed bin thresholds.
    """
    
    def __init__(self, num_bins: int = 1000, min_val: float = -10.0, max_val: float = 10.0):
        """
        Initialize the binary quantizer.
        
        Args:
            num_bins: Number of quantization bins
            min_val: Minimum value for quantization range
            max_val: Maximum value for quantization range
        """
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        
        # Compute quantization parameters (stateless - same for all samples)
        self.params = compute_binary_quantization_params(num_bins, min_val, max_val)
        
        logger.debug(f'BinaryQuantizer initialized - num_bins: {num_bins}, '
                    f'min_val: {min_val}, max_val: {max_val}')
    
    def fit(self, data: torch.Tensor) -> 'BinaryQuantizer':
        """Compatibility method - returns self since we're stateless."""
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data to binary quantized representation.
        
        Args:
            data: Input tensor of shape (batch_size, seq_len, features) or (seq_len, features)
            
        Returns:
            Binary tensor with additional dimension for bins
        """
        # Handle single sample case
        if data.dim() == 2:  # (seq_len, features)
            data = data.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
        
        # Apply binary quantization
        transformed = transform_binary_quantization_batch(data, self.params)
        
        if single_sample:
            transformed = transformed.squeeze(0)  # Remove batch dimension
        
        return transformed
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform binary quantized data back to continuous values.
        
        Args:
            data: Binary tensor with last dimension = num_bins
            
        Returns:
            Reconstructed continuous tensor
        """
        # Handle single sample case
        if data.dim() == 3:  # (seq_len, features, num_bins)
            data = data.unsqueeze(0)  # Add batch dimension
            single_sample = True
        else:
            single_sample = False
        
        # Apply inverse binary quantization
        reconstructed = inverse_transform_binary_quantization_batch(data, self.params)
        
        if single_sample:
            reconstructed = reconstructed.squeeze(0)  # Remove batch dimension
        
        return reconstructed


class BinScaler:
    """
    Combined scaler and quantizer for foundation models.
    
    This class combines a scaler (StandardScaler or TemporalScaler) with 
    BinaryQuantizer to provide the complete preprocessing pipeline expected by BinConv.
    """
    
    def __init__(self, scaler: Union[StandardScaler, TemporalScaler], quantizer: BinaryQuantizer):
        """
        Initialize the combined scaler and quantizer.
        
        Args:
            scaler: Either StandardScaler or TemporalScaler instance
            quantizer: BinaryQuantizer instance
        """
        self.scaler = scaler
        self.quantizer = quantizer
        
        logger.debug(f'BinScaler initialized with {type(scaler).__name__} and {type(quantizer).__name__}')
    
    def fit(self, data: torch.Tensor) -> 'BinScaler':
        """
        Fit both scaler and quantizer (stateless - returns self).
        
        Args:
            data: Input tensor to fit on
            
        Returns:
            self for method chaining
        """
        self.scaler.fit(data)
        self.quantizer.fit(data)
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply scaling followed by quantization.
        
        Args:
            data: Input tensor to transform
            
        Returns:
            Quantized tensor with binary representation
        """
        # First apply scaling
        scaled_data = self.scaler.transform(data)
        
        # Then apply quantization
        quantized_data = self.quantizer.transform(scaled_data)
        
        return quantized_data
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse quantization followed by inverse scaling.
        
        Args:
            data: Quantized tensor to inverse transform
            
        Returns:
            Original scale tensor
        """
        # First apply inverse quantization
        reconstructed_data = self.quantizer.inverse_transform(data)
        
        # Then apply inverse scaling (with warning about stateless limitation)
        original_data = self.scaler.inverse_transform(reconstructed_data)
        
        return original_data


# =============================================================================
# Utility Functions
# =============================================================================

def create_foundation_model_preprocessor(scaler_type: str, quantizer_params: Dict[str, Any]) -> BinScaler:
    """
    Create a preprocessor suitable for foundation models.
    
    Args:
        scaler_type: Type of scaler ('standard' or 'temporal')
        quantizer_params: Parameters for the quantizer (num_bins, min_val, max_val)
        
    Returns:
        BinScaler instance ready for foundation model preprocessing
    """
    # Create scaler based on type
    if scaler_type == 'standard':
        scaler = StandardScaler(var_specific=True)
    elif scaler_type == 'temporal':
        scaler = TemporalScaler(time_first=False)
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}. Must be 'standard' or 'temporal'")
    
    # Create quantizer
    quantizer = BinaryQuantizer(**quantizer_params)
    
    # Combine into BinScaler
    preprocessor = BinScaler(scaler, quantizer)
    
    logger.info(f'Created foundation model preprocessor with {scaler_type} scaler and binary quantizer')
    
    return preprocessor