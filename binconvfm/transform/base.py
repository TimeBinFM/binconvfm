"""
Base classes for stateless transformations in foundation time series models.

This module provides the core interface for stateless preprocessing transformations.
The design follows these principles:

1. fit() methods return parameters instead of storing state
2. transform() applies transformation and returns (data, params)
3. inverse_transform() uses provided parameters
4. Each sample in a batch is processed independently

This approach enables efficient per-sample preprocessing for foundation models
where cross-sample information leakage should be avoided.
"""

import torch
import logging
from typing import Dict, Tuple
from abc import ABC, abstractmethod

# Set up logger
logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """
    Abstract base class for stateless transformations.

    This class defines the standard interface for all transformation classes
    in the foundation time series preprocessing pipeline. All subclasses
    must implement the fit, transform, and inverse_transform methods.
    """

    def __init__(self):
        """Initialize base transform. Subclasses should set input dimensions."""
        self.input_dims = None
        self.inverse_input_dims = None

    def _get_shape_description(self, dims: int) -> str:
        """
        Get human-readable shape description for given dimensions.
        Subclasses should override this method to provide specific descriptions.

        Args:
            dims: Number of dimensions

        Returns:
            Human-readable description of expected shape
        """
        return f"{dims}D tensor"

    def _validate_input_shape(
        self, data: torch.Tensor, expected_dims: int, operation: str = "transform"
    ) -> None:
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
            raise ValueError(
                f"{self.__class__.__name__}.{operation}() expected {expected_dims}D tensor "
                f"with shape {expected_shape}, but got {data.dim()}D tensor with shape {data.shape}"
            )

    @abstractmethod
    def fit(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute transformation parameters from data.

        This method computes parameters needed for the transformation without
        storing them in the object state, maintaining statelessness.

        Args:
            data: Input tensor to compute parameters from

        Returns:
            Dictionary containing transformation parameters
        """
        raise NotImplementedError("Subclasses must implement fit method")

    @abstractmethod
    def transform(
        self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply transformation to data.

        Args:
            data: Input tensor to transform
            params: Pre-computed parameters. If None, will compute new ones via fit().
                   To avoid accidental parameter mismatch, this should be explicitly provided
                   when using pre-computed parameters.

        Returns:
            transformed_data
        """
        raise NotImplementedError("Subclasses must implement transform method")

    @abstractmethod
    def inverse_transform(
        self, data: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply inverse transformation to data.

        Args:
            data: Transformed tensor to inverse transform
            params: Parameters from transform() method

        Returns:
            iversed_tensor
        """
        raise NotImplementedError("Subclasses must implement inverse_transform method")

    def fit_transform(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fit transformation parameters and apply transformation.

        This is a convenience method equivalent to calling fit() and then transform()
        with the computed parameters.

        Args:
            data: Input tensor to fit and transform

        Returns:
            Tuple of (transformed_data, parameters)
        """
        params = self.fit(data)
        transformed_data = self.transform(data, params)
        return transformed_data, params
