"""Linear trend synthetic dataset generator."""

import torch
from typing import Iterator, Optional
from ..common import TensorIterableDataset


class LinearTrendDataset(TensorIterableDataset):
    """Synthetic dataset that generates linear trends with gaussian noise."""
    
    def __init__(
        self, 
        sequence_length: int = 100,
        num_sequences: int = 1000,
        trend_slope: float = 0.1,
        noise_std: float = 0.1,
        intercept: float = 0.0,
        seed: Optional[int] = None
    ):
        """Initialize the synthetic dataset.
        
        Args:
            sequence_length: Length of each time series
            num_sequences: Number of sequences to generate
            trend_slope: Slope of the linear trend
            noise_std: Standard deviation of gaussian noise
            intercept: Y-intercept of the linear trend
            seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.trend_slope = trend_slope
        self.noise_std = noise_std
        self.intercept = intercept
        
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Generate synthetic time series with linear trend and gaussian noise."""
        for i in range(self.num_sequences):
            # Generate time steps
            time_steps = torch.arange(self.sequence_length, dtype=torch.float32)
            
            # Linear trend
            trend = self.intercept + self.trend_slope * time_steps
            
            # Add gaussian noise
            noise = torch.normal(
                mean=0.0, 
                std=self.noise_std, 
                size=(self.sequence_length,),
                generator=self.generator
            )
            
            # Combine trend and noise
            series = trend + noise
            
            # Reshape to (sequence_length, 1) to match expected format
            yield series.unsqueeze(-1)
