"""
Quantization transformations for stateless preprocessing in foundation time series models.

This module provides quantization transformations that convert continuous values
to discrete representations. These are particularly useful for foundation models
that operate on quantized time series data.

Available quantizers:
- BinaryQuantizer: Converts continuous values to binary vectors
"""

import torch
import logging
from typing import Dict, Tuple

from .base import BaseTransform

# Set up logger
logger = logging.getLogger(__name__)


class BinaryQuantizer(BaseTransform):
    """
    Stateless binary quantizer for foundation models.

    Transforms continuous values into binary vectors using fixed bin thresholds.
    Each continuous value is converted to a binary vector where each element
    indicates whether the value exceeds the corresponding bin threshold.

    The quantization parameters (bin edges and values) are stateful class attributes
    since they should be consistent across all samples and calls. This ensures
    that the same continuous value always maps to the same binary representation.

    The binary vectors have a monotonic structure: all 1s followed by all 0s,
    which represents "value >= threshold" for each threshold.
    """

    def __init__(
        self, num_bins: int = 1000, min_val: float = -10.0, max_val: float = 10.0
    ):
        """
        Initialize the binary quantizer.

        Args:
            num_bins: Number of quantization bins
            min_val: Minimum value for quantization range
            max_val: Maximum value for quantization range
        """
        super().__init__()
        self.input_dims = 3  # (batch_size, context_length, n_features)
        self.inverse_input_dims = (
            4  # (batch_size, context_length, n_features, num_bins)
        )

        # Store quantization configuration
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Compute quantization parameters (stateful - same for all samples)
        self.bin_edges = torch.linspace(min_val, max_val, num_bins + 1)
        self.bin_values = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        logger.debug(
            f"BinaryQuantizer initialized - num_bins: {num_bins}, "
            f"min_val: {min_val}, max_val: {max_val}"
        )

    def _get_shape_description(self, dims: int) -> str:
        """Get shape description specific to BinaryQuantizer."""
        if dims == 3:
            return "(batch_size, context_length, n_features)"
        elif dims == 4:
            return "(batch_size, context_length, n_features, num_bins)"
        else:
            return f"{dims}D tensor"

    def fit(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute quantization parameters from data.

        For BinaryQuantizer, the quantization parameters are fixed (stored as
        class attributes), so this method returns an empty dictionary for
        consistency with the interface.

        Args:
            data: Input tensor (used for validation only)

        Returns:
            Empty dictionary (parameters are stored as class attributes)
        """
        self._validate_input_shape(data, self.input_dims, "fit")

        # Return empty dict since quantization parameters are stored in the object
        return {}

    def transform(
        self, data: torch.Tensor, params: Dict[str, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transform continuous data to binary quantized representation.

        Args:
            data: Input tensor of shape (batch_size, context_length, n_features)
            params: Optional pre-computed parameters. For BinaryQuantizer, this is
                   ignored since quantization parameters are stored in the object.

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
        # Each element is 1 if data >= threshold, 0 otherwise
        binary_vectors = (data_expanded >= bin_thresholds).float()

        return binary_vectors

    def inverse_transform(
        self, data: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Inverse transform binary quantized data back to continuous values.

        This method reconstructs continuous values from binary vectors by finding
        the highest active bin and using its corresponding bin value.

        Args:
            data: Binary tensor of shape (batch_size, context_length, n_features, num_bins)
            params: Empty params dict (not used, kept for interface consistency)

        Returns:
            Tuple of (reconstructed_tensor, parameters_used)
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
        # Use the lowest bin value for samples with no active bins
        all_zero_mask = data.sum(dim=-1) == 0
        reconstructed = torch.where(all_zero_mask, bin_values[0], reconstructed)

        return reconstructed
