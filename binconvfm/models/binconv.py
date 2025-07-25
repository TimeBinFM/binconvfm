"""
BinConv: Binary Convolution Foundation Model for Time Series Forecasting.

This module implements the BinConv architecture, a foundation model for time series
forecasting that uses binary quantization combined with 2D and 1D convolutions to
perform autoregressive forecasting with per-sample preprocessing.

Key Features:
- Foundation model approach with per-sample preprocessing
- Binary quantization for efficient representation
- 2D and 1D convolutional blocks with residual connections
- PyTorch Lightning integration for scalable training
- Stateless preprocessing for consistent inference

Author: BinConvFM Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Literal, Union, Optional, Dict, Tuple, List
import warnings

# PyTorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

# Local imports
from binconvfm.layers.DynamicTanh import DynamicTanh
from binconvfm.utils.processing import (
    StandardScaler, TemporalScaler, BinaryQuantizer, TransformPipeline
)
from binconvfm.utils.forecast import get_sequence_from_prob, most_probable_monotonic_sequence
from binconvfm.utils.reshape import repeat

# Set up logger
logger = logging.getLogger(__name__)


class BinConv(nn.Module):
    """
    Binary Convolution model for time series forecasting.
    
    This model uses binary quantization combined with 2D and 1D convolutions
    to perform autoregressive forecasting. The architecture processes quantized
    time series data through multiple convolutional blocks with residual connections.
    
    The model is designed as a foundation model, meaning each time series sample
    is preprocessed independently without cross-sample information leakage.
    """
    
    def __init__(
        self, 
        context_length: int, 
        is_prob_forecast: bool, 
        num_bins: int, 
        min_bin_value: float = -10.0,
        max_bin_value: float = 10.0, 
        kernel_size_across_bins_2d: int = 3,
        kernel_size_across_bins_1d: int = 3, 
        num_filters_2d: int = 8,
        num_filters_1d: int = 32, 
        is_cum_sum: bool = False, 
        num_1d_layers: int = 2, 
        num_blocks: int = 3,
        kernel_size_ffn: int = 51, 
        dropout: float = 0.2,
        last_layer: Literal["conv", "fc"] = 'conv',
        scaler_type: Union[Literal["standard", "temporal", "None"], None] = None
    ) -> None:
        """
        Initialize the BinConv model.
        
        Args:
            context_length: Length of input context window
            is_prob_forecast: Whether to use probabilistic forecasting
            num_bins: Number of quantization bins
            min_bin_value: Minimum value for quantization range
            max_bin_value: Maximum value for quantization range
            kernel_size_across_bins_2d: Kernel size for 2D convolution across bins
            kernel_size_across_bins_1d: Kernel size for 1D convolution across bins
            num_filters_2d: Number of filters for 2D convolution
            num_filters_1d: Number of filters for 1D convolution
            is_cum_sum: Whether to apply cumulative sum (deprecated, set to False)
            num_1d_layers: Number of 1D convolutional layers
            num_blocks: Number of convolutional blocks
            kernel_size_ffn: Kernel size for feed-forward network
            dropout: Dropout rate
            last_layer: Type of last layer ("conv" or "fc")
            scaler_type: Type of scaler ("standard", "temporal", or None)
            
        Raises:
            AssertionError: If kernel sizes are not odd
            ValueError: If unsupported scaler_type is provided
        """
        super().__init__()
        
        # Validate inputs
        if kernel_size_across_bins_2d % 2 == 0:
            raise AssertionError("2D kernel size must be odd")
        if kernel_size_across_bins_1d % 2 == 0:
            raise AssertionError("1D kernel size must be odd")
            
        # Store configuration
        self.context_length = context_length
        self.num_bins = num_bins
        self.min_bin_value = min_bin_value
        self.max_bin_value = max_bin_value
        self.is_prob_forecast = is_prob_forecast
        self.num_filters_2d = num_filters_2d
        self.num_filters_1d = num_filters_1d
        self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
        self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
        self.is_cum_sum = is_cum_sum
        self.num_1d_layers = num_1d_layers
        self.num_blocks = num_blocks
        self.kernel_size_ffn = kernel_size_ffn
        self.last_layer = last_layer
        
        logger.info(f'BinConv initialized - dropout: {dropout}')
        
        # Initialize preprocessing scaler for foundation model approach
        self._initialize_scaler(scaler_type)
        
        # Initialize dropout
        self.dropout = nn.Dropout(dropout)
        
        # Validate filter dimensions
        if num_filters_2d != num_filters_1d:
            raise ValueError("num_filters_2d must equal num_filters_1d (architectural constraint)")
            
        # Initialize activation functions
        self._initialize_activations()
        
        # Initialize convolutional layers
        self._initialize_conv_layers()
        
        logger.info(f'BinConv model initialized with {sum(p.numel() for p in self.parameters())} parameters')
    
    def _initialize_scaler(self, scaler_type: Union[str, None]) -> None:
        """
        Initialize preprocessing scaler for foundation model approach.
        
        Args:
            scaler_type: Type of scaler to use
        """
        logger.info(f'Per-sample scaler type: {scaler_type}')
        
        if scaler_type is None:
            self.scaler = None
            logger.info('No preprocessors initialized (raw data mode)')
        elif scaler_type == 'standard':
            self.scaler = TransformPipeline([
                    ('scaler', StandardScaler()),
                    ('quantizer', BinaryQuantizer(
                        num_bins=self.num_bins, 
                        min_val=self.min_bin_value, 
                        max_val=self.max_bin_value
                    ))
                ])

            logger.info(f'Initialized a standard scaling preprocessor')
        elif scaler_type == 'temporal':
            self.scaler = TransformPipeline([
                    ('scaler', TemporalScaler()),
                    ('quantizer', BinaryQuantizer(
                        num_bins=self.num_bins, 
                        min_val=self.min_bin_value, 
                        max_val=self.max_bin_value
                    ))
                ])
            logger.info(f'Initialized a temporal scaling preprocessors')
        else:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}. "
                           f"Must be 'standard', 'temporal', or None.")
    
    def _initialize_activations(self) -> None:
        """Initialize activation functions for convolutional blocks."""
        self.act = nn.ModuleList([
            nn.ModuleList([
                DynamicTanh(
                    normalized_shape=self.num_filters_2d if i < self.num_1d_layers else self.context_length,
                    channels_last=False
                )
                for i in range(1)  # Applied only after conv2d
            ]) for _ in range(self.num_blocks)
        ])
        
        logger.debug(f'Initialized activation functions: {len(self.act)} blocks')
    
    def _initialize_conv_layers(self) -> None:
        """Initialize convolutional layers for single target dimension."""
        # 2D Convolutional layers
        conv2d = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=self.num_filters_2d,
                kernel_size=(self.context_length, self.kernel_size_across_bins_2d),
                bias=True
            ) for _ in range(self.num_blocks)
        ])
        
        # 1D Convolutional layers
        conv1d = nn.ModuleList([
            nn.ModuleList([
                nn.Conv1d(
                    in_channels=self.num_filters_2d if i == 0 else self.num_filters_1d,
                    out_channels=self.context_length if i == self.num_1d_layers - 1 else self.num_filters_1d,
                    kernel_size=self.kernel_size_across_bins_1d, 
                    bias=True,
                    groups=self.num_filters_1d
                ) for i in range(self.num_1d_layers)
            ]) for _ in range(self.num_blocks)
        ])
        
        # Feed-forward network layer
        if self.last_layer == 'conv':
            conv_ffn = nn.Conv1d(
                in_channels=self.context_length,
                out_channels=1,
                kernel_size=self.kernel_size_ffn,
                groups=1,
                bias=True
            )
        elif self.last_layer == 'fc':
            class MeanOverChannel(nn.Module):
                def __init__(self, dim: int = -2):
                    super().__init__()
                    self.dim = dim

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    return x.mean(dim=self.dim)

            conv_ffn = nn.Sequential(
                MeanOverChannel(dim=-2),
                nn.Linear(in_features=self.num_bins, out_features=self.num_bins, bias=True)
            )
        else:
            raise ValueError(f"Unsupported last_layer type: {self.last_layer}")
        
        # Activation functions
        act = nn.ModuleList([
            nn.ModuleList([
                DynamicTanh(
                    normalized_shape=self.num_filters_2d if i < self.num_1d_layers else self.context_length,
                    channels_last=False
                ) for i in range(1)
            ]) for _ in range(self.num_blocks)
        ])
        
        logger.debug('Initialized layers for single target dimension:')
        logger.debug(f'  - Conv2D blocks: {len(conv2d)}')
        logger.debug(f'  - Conv1D blocks: {len(conv1d)}')
        logger.debug(f'  - FFN layer type: {self.last_layer}')
        
        self.layers = nn.ModuleDict({
            'conv2d': conv2d,
            'conv1d': conv1d,
            'conv_ffn': conv_ffn,
            'act': act,
        })
    
    @staticmethod
    def _pad_channels(
        tensor: torch.Tensor, 
        pad_size: int, 
        pad_val_left: float = 1.0, 
        pad_val_right: float = 0.0
    ) -> torch.Tensor:
        """
        Pad tensor channels for convolution.
        
        Args:
            tensor: Input tensor to pad
            pad_size: Size of padding
            pad_val_left: Value for left padding
            pad_val_right: Value for right padding
            
        Returns:
            Padded tensor
        """
        if pad_size == 0:
            return tensor
            
        left = torch.full((*tensor.shape[:-1], pad_size), pad_val_left, device=tensor.device)
        right = torch.full((*tensor.shape[:-1], pad_size), pad_val_right, device=tensor.device)
        
        return torch.cat([left, tensor, right], dim=-1)
    
    def conv_layer(
        self, 
        x: torch.Tensor, 
        conv_func: nn.Module, 
        act_func: Optional[nn.Module], 
        kernel_size: int, 
        is_2d: bool
    ) -> torch.Tensor:
        """
        Apply convolution layer with padding and activation.
        
        Args:
            x: Input tensor
            conv_func: Convolution function to apply
            act_func: Activation function to apply (optional)
            kernel_size: Size of convolution kernel
            is_2d: Whether this is a 2D convolution
            
        Returns:
            Processed tensor after convolution and activation
        """
        # Calculate padding
        pad = kernel_size // 2 if kernel_size > 1 else 0
        x_padded = self._pad_channels(x, pad)
        
        # Add channel dimension for 2D convolution
        if is_2d:
            x_padded = x_padded.unsqueeze(1)
        
        # Apply convolution
        conv_out = conv_func(x_padded)
        
        # Remove spatial dimension for 2D convolution
        if is_2d:
            conv_out = conv_out.squeeze(2)
        
        # Apply activation if provided
        if act_func is not None:
            conv_out = act_func(conv_out)
        
        return conv_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the BinConv model.
        
        Args:
            x: Input tensor of shape (batch_size, context_length, num_bins)
            
        Returns:
            Output tensor of shape (batch_size, num_bins)
            
        Raises:
            AssertionError: If input context length doesn't match expected length
        """
        
        x = x.float()
        batch_size, context_length, num_bins = x.shape
        
        if context_length != self.context_length:
            raise AssertionError(f"Input context length {context_length} doesn't match "
                               f"expected length {self.context_length}")
        
        logger.debug(f'Forward pass - input shape: {x.shape}')
        
        # Process through convolutional blocks with residual connections
        for block_idx in range(self.num_blocks):
            residual = x
            
            # 2D Convolution
            x = self.conv_layer(
                x, 
                self.layers["conv2d"][block_idx], 
                self.layers["act"][block_idx][0],
                self.kernel_size_across_bins_2d, 
                is_2d=True
            )
            
            # 1D Convolutions
            for layer_idx in range(self.num_1d_layers):
                x = self.conv_layer(
                    x, 
                    self.layers["conv1d"][block_idx][layer_idx], 
                    F.relu,
                    self.kernel_size_across_bins_1d, 
                    is_2d=False
                )
            
            # Apply dropout and residual connection
            x = self.dropout(x)
            x = x + residual
        
        # Final feed-forward layer
        if self.last_layer == 'conv':
            out = self.conv_layer(
                x, 
                self.layers["conv_ffn"], 
                None, 
                self.kernel_size_ffn, 
                is_2d=False
            ).squeeze(1)
        else:
            out = self.layers["conv_ffn"](x)
        
        # Apply cumulative sum if enabled (deprecated)
        if self.is_cum_sum:
            raise NotImplementedError("Cumulative sum is disabled as it degrades performance")
        
        return out
    
    @torch.inference_mode()
    def forecast(
        self, 
        batch_data: torch.Tensor,
        prediction_length: Optional[int],
        num_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate forecasts for input batch using autoregressive prediction.
        
        This method applies per-sample preprocessing and generates forecasts
        independently for each sample, maintaining the foundation model philosophy.
        
        Args:
            batch_data: Input tensor of shape (batch_size, context_length, 1)
            num_samples: Number of forecast samples to generate (for probabilistic forecasting)
            
        Returns:
            Forecast tensor with shape depending on sampling:
            - If num_samples > 1: (batch_size, num_samples, prediction_length)
            - Otherwise: (batch_size, 1, prediction_length)
        """
        do_sample = num_samples is not None and num_samples > 1 and self.is_prob_forecast
        if self.scaler is not None:
            c_inputs, current_pipeline_params = self.scaler.transform(batch_data)
            c_inputs = c_inputs.squeeze(-2)
        else:
            c_inputs = batch_data
            current_pipeline_params = None

        if do_sample:
            c_inputs = repeat(c_inputs.unsqueeze(1), num_samples, 1)  # (B, NS, T, D)
            batch_size = c_inputs.shape[0]
            c_inputs = c_inputs.view(-1, *c_inputs.shape[2:]) # to process using only one inference

        current_context = c_inputs.clone()
        c_forecasts = []
        for step in range(prediction_length):
            # Forward pass
            pred = F.sigmoid(self(current_context))  # (B, num_bins)

            # Sample or take most probable sequence
            pred, _ = get_sequence_from_prob(pred, do_sample)
            pred = pred.int().unsqueeze(1) # (B, 1, num_bins)
            c_forecasts.append(pred)

            # Update context for next step
            next_input = pred
            current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

        # Concatenate forecasts
        c_forecasts = torch.cat(c_forecasts, dim=1)  # (B, prediction_length, num_bins)

        # Apply inverse transformation
        if self.scaler is not None and current_pipeline_params is not None:
            c_forecasts = self.scaler.inverse_transform(c_forecasts.unsqueeze(0), current_pipeline_params)

        # Reshape for sampling
        if do_sample:
            c_forecasts = c_forecasts.view(batch_size, num_samples, *c_forecasts.shape[1:])
        else:
            c_forecasts = c_forecasts.unsqueeze(1)  # (B, 1, T, D)

        return c_forecasts


class LightningBinConv(BinConv, LightningModule):
    """
    PyTorch Lightning wrapper for BinConv model.
    
    Provides training loop, loss computation, and optimizer configuration
    for the binary convolution forecasting model. Handles preprocessing
    automatically and supports both training and validation.
    """
    
    def __init__(self, lr: float = 1e-3, *args, **kwargs):
        """
        Initialize Lightning wrapper for BinConv.
        
        Args:
            lr: Learning rate for optimizer
            *args: Arguments passed to BinConv
            **kwargs: Keyword arguments passed to BinConv
        """
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.save_hyperparameters()
        
        logger.info(f'LightningBinConv initialized with learning rate: {lr}')
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step for one batch.
        
        Args:
            batch: Tuple of (inputs, targets) tensors where:
                   - inputs: Raw input data (batch_size, context_length, target_dim)
                   - targets: Target data (batch_size, prediction_length, target_dim)
            batch_idx: Index of the current batch
            
        Returns:
            torch.Tensor: Training loss for this batch
        """
        inputs, targets = batch


        if self.scaler is not None:
            inputs, params = self.scaler.transform(inputs)
            targets, _ = self.scaler.transform(targets.unsqueeze(-1), params)
            inputs, targets = inputs.squeeze(), targets.squeeze()

        logits = self(inputs)

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx % 100 == 0:  # Log every 100 batches to avoid spam
            logger.debug(f'Batch {batch_idx}: Training loss = {loss.item():.6f}')
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Validation step for one batch.
        
        Args:
            batch: Tuple of (inputs, targets) tensors
            batch_idx: Index of the current batch
            
        Returns:
            torch.Tensor: Validation loss for this batch
        """
        inputs, targets = batch

        if self.scaler is not None:
            inputs, params = self.scaler.transform(inputs)
            targets, _ = self.scaler.transform(targets.unsqueeze(-1), params)
            inputs, targets = inputs.squeeze(), targets.squeeze()

        logits = self(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with specified learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        logger.info(f'Configured Adam optimizer with lr={self.lr}')
        return optimizer
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup method called by Lightning.
        
        Args:
            stage: Training stage ('fit', 'validate', 'test', 'predict')
        """
        logger.info(f'Lightning setup called for stage: {stage}')
        logger.info('BinConv uses per-sample preprocessing (foundation model approach)')