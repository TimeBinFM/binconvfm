"""
Scaling transformations for stateless preprocessing in foundation time series models.

This module provides scaling transformations that compute parameters per sample
and apply transformations independently, making them ideal for foundation models
where each time series should be processed without cross-sample information leakage.

Available scalers:
- StandardScaler: Per-sample standardization (zero mean, unit variance)
- TemporalScaler: Per-sample temporal scaling (divide by mean)
"""

import torch
import logging
from typing import Dict, Tuple

from .base import BaseTransform

# Set up logger
logger = logging.getLogger(__name__)


class IdentityTransform(BaseTransform):
    """
    Identity transformation that returns data unchanged.
    
    This transform is useful as a placeholder or for pipelines where 
    no transformation is needed. All methods return the data unchanged
    while maintaining the stateless interface.
    """
    
    def __init__(self):
        """Initialize the identity transform."""
        super().__init__()
        self.input_dims = None  # Accept any number of dimensions
        self.inverse_input_dims = None  # Accept any number of dimensions
        logger.debug('IdentityTransform initialized')
    
    def fit(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Fit method for identity transform (returns empty parameters).
        
        Args:
            data: Input tensor (unused, kept for interface consistency)
            
        Returns:
            Empty dictionary
        """
        return {}
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Transform data (returns unchanged).
        
        Args:
            data: Input tensor to "transform"
            params: Optional parameters (ignored)
            
        Returns:
            Tuple of (unchanged_data, empty_params)
        """
        return data, {}
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inverse transform data (returns unchanged).
        
        Args:
            data: Input tensor to "inverse transform"
            params: Parameters (ignored)
            
        Returns:
            Tuple of (unchanged_data, parameters_used)
        """
        return data, params


class StandardScaler(BaseTransform):
    """
    Stateless standard scaler for foundation models.
    
    Computes standardization parameters per sample (zero mean, unit variance)
    and applies them independently for each sample in the batch. This maintains
    the per-sample independence required for foundation models.
    
    The transformation is: (x - mean) / std
    where mean and std are computed per sample across the time dimension.
    """
    
    def __init__(self):
        """Initialize the standard scaler."""
        super().__init__()
        self.input_dims = 3  # (batch_size, context_length, n_features)
        self.inverse_input_dims = None  # Accept 3D or 4D tensors for inverse transform
        logger.debug('StandardScaler initialized for foundation model use')
    
    def _get_shape_description(self, dims: int) -> str:
        """Get shape description specific to StandardScaler."""
        if dims == 3:
            return "(batch_size, context_length, n_features)"
        else:
            return f"{dims}D tensor"
    
    def fit(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute standardization parameters from data.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            
        Returns:
            Dictionary with 'mean' and 'std' tensors of shape (batch_size, 1, n_features)
        """
        self._validate_input_shape(data, self.input_dims, "fit")
        
        # Compute per-sample statistics across time dimension
        mean = data.mean(dim=1, keepdim=True)  # (batch_size, 1, n_features)
        std = data.std(dim=1, keepdim=True, unbiased=False)  # (batch_size, 1, n_features)
        
        # Prevent division by zero by adding appropriate epsilon
        eps = torch.finfo(data.dtype).eps
        std = std + eps
        
        return {'mean': mean, 'std': std}
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply standardization transformation to data.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            params: Optional pre-computed parameters. If provided, uses these instead
                   of computing new ones. Should contain 'mean' and 'std' keys.
            
        Returns:
            Tuple of (standardized_data, parameters_used)
        """
        self._validate_input_shape(data, self.input_dims, "transform")
        
        if params is not None:
            # Use provided parameters
            mean = params['mean'].to(data.device)
            std = params['std'].to(data.device)
            result_params = params
        else:
            # Compute new parameters
            result_params = self.fit(data)
            mean = result_params['mean']
            std = result_params['std']
        
        # Apply standardization
        transformed = (data - mean) / std
        
        return transformed, result_params
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inverse standardization transformation.
        
        Args:
            data: Standardized tensor of shape (batch_size, context_length, n_features)
                  or (batch_size, n_samples, horizon, n_features)
            params: Parameters from transform() method containing 'mean' and 'std'
            
        Returns:
            Tuple of (original_scale_tensor, parameters_used)
        """
        # Validate that data is either 3D or 4D
        if data.dim() not in [3, 4]:
            raise ValueError(f"StandardScaler.inverse_transform() expected 3D or 4D tensor, "
                           f"but got {data.dim()}D tensor with shape {data.shape}")
        
        mean = params['mean'].to(data.device)
        std = params['std'].to(data.device)
        
        # Handle both 3D and 4D tensors
        if data.dim() == 4:
            # For 4D tensors (batch, n_samples, horizon, features), 
            # expand mean and std to match: (batch, 1, 1, features)
            mean = mean.unsqueeze(1)  # (batch, 1, 1, features)
            std = std.unsqueeze(1)    # (batch, 1, 1, features)
        
        # Apply inverse transformation: x = (standardized * std) + mean
        original_data = data * std + mean
        
        return original_data, params


class TemporalScaler(BaseTransform):
    """
    Stateless temporal scaler for foundation models.
    
    Scales data by dividing by the per-sample mean, preserving relative temporal 
    patterns while normalizing the scale. This is useful for preserving the
    temporal dynamics while reducing scale differences between samples.
    
    The transformation is: x / abs(mean)
    where mean is computed per sample across the time dimension.
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
    
    def fit(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute temporal scaling parameters from data.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            
        Returns:
            Dictionary with 'mean' tensor of shape (batch_size, 1, n_features)
        """
        self._validate_input_shape(data, self.input_dims, "fit")
        
        # Compute per-sample mean across time dimension
        mean = data.mean(dim=1, keepdim=True)  # (batch_size, 1, n_features)
        
        # Take absolute value and add epsilon to prevent division by zero
        eps = torch.finfo(data.dtype).eps
        mean = torch.abs(mean) + eps
        
        return {'mean': mean}
    
    def transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply temporal scaling transformation to data.
        
        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            params: Optional pre-computed parameters. If provided, uses these instead
                   of computing new ones. Should contain 'mean' key.
            
        Returns:
            Tuple of (scaled_data, parameters_used)
        """
        self._validate_input_shape(data, self.input_dims, "transform")
        
        if params is not None:
            # Use provided parameters
            mean = params['mean'].to(data.device)
            result_params = params
        else:
            # Compute new parameters
            result_params = self.fit(data)
            mean = result_params['mean']
        
        # Apply temporal scaling
        transformed = data / mean
        
        return transformed, result_params
    
    def inverse_transform(self, data: torch.Tensor, params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Inverse temporal scaling transformation.
        
        Args:
            data: Scaled tensor of shape (batch_size, context_length, n_features)
            params: Parameters from transform() method containing 'mean'
            
        Returns:
            Tuple of (original_scale_tensor, parameters_used)
        """
        self._validate_input_shape(data, self.inverse_input_dims, "inverse_transform")
        
        mean = params['mean'].to(data.device)
        
        # Apply inverse transformation: x = scaled * mean
        original_data = data * mean
        
        return original_data, params