"""
Transform module for stateless preprocessing in foundation time series models.

This module provides a sklearn-like interface for torch tensors with stateless transforms.
All fit methods return parameters instead of storing them, enabling per-sample processing
for foundation models.

The interface follows sklearn conventions:
- fit(X) returns parameters
- transform(X, params=None) applies transformation and returns (transformed_X, params)
- inverse_transform(X, params) applies inverse transformation
- Pipeline chains multiple transforms

Example usage:
    from binconvfm.transform import StandardScaler, BinaryQuantizer, Pipeline

    # Create transforms
    scaler = StandardScaler()
    quantizer = BinaryQuantizer(num_bins=1000)

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('quantizer', quantizer)
    ])

    # Use pipeline
    transformed_data, params = pipeline.transform(data)
    original_data = pipeline.inverse_transform(transformed_data, params)
"""

from .base import BaseTransform
from .scalers import IdentityTransform, StandardScaler, TemporalScaler
from .quantizers import BinaryQuantizer
from .pipeline import Pipeline

__all__ = [
    "BaseTransform",
    "IdentityTransform",
    "StandardScaler",
    "TemporalScaler",
    "BinaryQuantizer",
    "Pipeline",
]
