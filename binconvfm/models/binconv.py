import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
from binconvfm.models.base import BaseForecaster, BaseLightningModule, BaseModel
from typing import Optional, List, Dict, Any
from binconvfm.utils.forecast import get_sequence_from_prob
import logging

# Local imports
from binconvfm.layers.DynamicTanh import DynamicTanh
from binconvfm.models.base import BaseForecaster, BaseLightningModule
from binconvfm.transform import BinaryQuantizer

# Set up logger
logger = logging.getLogger(__name__)


class BinConvForecaster(BaseForecaster):
    """
    BinConv forecaster following the standard interface.
    """

    def __init__(
            self,
            # Common parameters - explicit
            horizon: int = 1,
            n_samples: int = 5,
            quantiles: List[float] = None,
            batch_size: int = 32,
            num_epochs: int = 10,
            lr: float = 0.001,
            accelerator: str = "cpu",
            enable_progress_bar: bool = True,
            logging: bool = False,
            log_every_n_steps: int = 10,
            transform: List[str] = None,
            transform_args: Optional[Dict[str, Dict[str, Any]]] = None,
            # BinConv-specific parameters - go to kwargs
            **kwargs
    ):
        """Initialize BinConvForecaster."""
        if quantiles is None:
            quantiles = [(i + 1) / 10 for i in range(9)]
        if transform is None:
            transform = ['StandardScaler', 'BinaryQuantizer']

        # Set default values for BinConv-specific parameters if not provided
        binconv_defaults = {
            'context_length': 512,
            'num_bins': 1024,
            'min_bin_value': -10.0,
            'max_bin_value': 10.0,
            'kernel_size_across_bins_2d': 3,
            'kernel_size_across_bins_1d': 3,
            'num_filters_2d': 32,
            'num_filters_1d': 32,
            'num_1d_layers': 2,
            'num_blocks': 3,
            'kernel_size_ffn': 51,
            'dropout': 0.2,
        }

        # Update defaults with provided kwargs
        for key, default_value in binconv_defaults.items():
            if key not in kwargs:
                kwargs[key] = default_value

        # Set up default transform_args if not provided and BinaryQuantizer is used
        if transform_args is None:
            transform_args = {}

        if 'BinaryQuantizer' in transform and 'BinaryQuantizer' not in transform_args:
            transform_args['BinaryQuantizer'] = {
                'num_bins': kwargs['num_bins'],
                'min_val': kwargs['min_bin_value'],
                'max_val': kwargs['max_bin_value']
            }

        super().__init__(
            horizon=horizon,
            n_samples=n_samples,
            quantiles=quantiles,
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            accelerator=accelerator,
            enable_progress_bar=enable_progress_bar,
            logging=logging,
            log_every_n_steps=log_every_n_steps,
            transform=transform,
            transform_args=transform_args,
            **kwargs  # Pass BinConv-specific parameters
        )

    def _create_model(self):
        """Create BinConvModule."""

        self.model = BinConvModule(
            horizon=self.horizon,
            n_samples=self.n_samples,
            quantiles=self.quantiles,
            lr=self.lr,
            transform=self.transform,
            transform_args=self.transform_args,
            context_length=self.kwargs['context_length'],
            num_bins=self.kwargs['num_bins'],
            min_bin_value=self.kwargs['min_bin_value'],
            max_bin_value=self.kwargs['max_bin_value'],
            kernel_size_across_bins_2d=self.kwargs['kernel_size_across_bins_2d'],
            kernel_size_across_bins_1d=self.kwargs['kernel_size_across_bins_1d'],
            num_filters_2d=self.kwargs['num_filters_2d'],
            num_filters_1d=self.kwargs['num_filters_1d'],
            num_1d_layers=self.kwargs['num_1d_layers'],
            num_blocks=self.kwargs['num_blocks'],
            kernel_size_ffn=self.kwargs['kernel_size_ffn'],
            dropout=self.kwargs['dropout'],
        )


class BinConvModule(BaseLightningModule):
    def __init__(
            self,
            context_length: int,
            num_bins: int,
            min_bin_value: float,
            max_bin_value: float,
            kernel_size_across_bins_2d: int,
            kernel_size_across_bins_1d: int,
            num_filters_2d: int,
            num_filters_1d: int,
            num_1d_layers: int,
            num_blocks: int,
            kernel_size_ffn: int,
            dropout: float,
            horizon: int,
            n_samples: int,
            quantiles: List[float],
            lr: float,
            transform: List[str],
            transform_args: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize BinConvModule."""
        super().__init__(n_samples, quantiles, lr, transform, transform_args)
        self.save_hyperparameters()
        # Create the BinConv model
        self.model = BinConv(
            context_length=context_length,
            num_bins=num_bins,
            min_bin_value=min_bin_value,
            max_bin_value=max_bin_value,
            kernel_size_across_bins_2d=kernel_size_across_bins_2d,
            kernel_size_across_bins_1d=kernel_size_across_bins_1d,
            num_filters_2d=num_filters_2d,
            num_filters_1d=num_filters_1d,
            num_1d_layers=num_1d_layers,
            num_blocks=num_blocks,
            kernel_size_ffn=kernel_size_ffn,
            dropout=dropout,
        )

    def configure_optimizers(self) -> Optimizer:
        """
        Configure and return the optimizer for training.

        Returns:
            Optimizer: The optimizer instance (Adam).
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def loss(self, output_seq: torch.Tensor, target_seq: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Compute loss for BinConv."""
        if target_seq.ndim == 4:
            target_seq = target_seq.squeeze(dim=(1, 2))
        else:
            assert target_seq.ndim == 2, "unexpected target sequence dimension"
        return F.binary_cross_entropy_with_logits(output_seq, target_seq)


class BinConv(BaseModel):
    """
    Binary Convolution model for time series forecasting.

    This model uses binary quantization combined with 2D and 1D convolutions
    to perform autoregressive forecasting. The architecture processes quantized
    time series data through multiple convolutional blocks with residual connections.
    """

    def __init__(
            self,
            context_length: int,
            num_bins: int = 1024,
            min_bin_value: float = -15.0,
            max_bin_value: float = 15.0,
            kernel_size_across_bins_2d: int = 3,
            kernel_size_across_bins_1d: int = 3,
            num_filters_2d: int = 32,
            num_filters_1d: int = 32,
            num_1d_layers: int = 2,
            num_blocks: int = 3,
            kernel_size_ffn: int = 51,
            dropout: float = 0.2,
            max_samples_per_batch: int = None,
    ) -> None:
        """
        Initialize the BinConv model.

        Args:
            context_length: Length of input context window
            num_bins: Number of quantization bins
            min_bin_value: Minimum value for quantization range
            max_bin_value: Maximum value for quantization range
            kernel_size_across_bins_2d: Kernel size for 2D convolution across bins
            kernel_size_across_bins_1d: Kernel size for 1D convolution across bins
            num_filters_2d: Number of filters for 2D convolution
            num_filters_1d: Number of filters for 1D convolution
            num_1d_layers: Number of 1D convolutional layers
            num_blocks: Number of convolutional blocks
            kernel_size_ffn: Kernel size for feed-forward network
            dropout: Dropout rate
            max_samples_per_batch: Number of maximum samples to process per batch (defaults to all samples)

        Raises:
            AssertionError: If kernel sizes are not odd
            ValueError: If filter dimensions don't match
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
        self.num_filters_2d = num_filters_2d
        self.num_filters_1d = num_filters_1d
        self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
        self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
        self.num_1d_layers = num_1d_layers
        self.num_blocks = num_blocks
        self.kernel_size_ffn = kernel_size_ffn
        self.max_samples_per_batch = max_samples_per_batch

        logger.info(f'BinConv initialized - dropout: {dropout}')

        # Initialize dropout
        self.dropout = nn.Dropout(dropout)

        # Validate filter dimensions
        if num_filters_2d != num_filters_1d:
            raise ValueError("num_filters_2d must equal num_filters_1d (architectural constraint)")

        # Initialize convolutional layers
        self._initialize_conv_layers()

        logger.info(f'BinConv model initialized with {sum(p.numel() for p in self.parameters())} parameters')

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

        # Feed-forward network layer (always convolutional)
        conv_ffn = nn.Conv1d(
            in_channels=self.context_length,
            out_channels=1,
            kernel_size=self.kernel_size_ffn,
            groups=1,
            bias=True
        )

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
        logger.debug('  - FFN layer type: conv')

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

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the BinConv model with sample batching.

        Args:
            x: Input tensor of shape (batch_size, context_length, num_bins)
            y: unused for this model
        Returns:
            - (batch_size, num_bins)
        Raises:
            AssertionError: If input context length doesn't match expected length
        """

        assert (
                x.shape[2] == 1
        ), "Input sequence must have a single feature dimension (dim=1)"
        x = x.squeeze(2).float()

        original_batch_size, context_length, num_bins_input = x.shape

        if context_length != self.context_length:
            raise AssertionError(f"Input context length {context_length} doesn't match "
                                 f"expected length {self.context_length}")

        return self._forward_single(x)

    def forecast(self, x: torch.Tensor, horizon: int, n_samples: int) -> torch.Tensor:
        """Generate autoregressive forecasts for multiple horizons."""

        max_samples_per_batch = self.max_samples_per_batch if  self.max_samples_per_batch else n_samples

        # Use deterministic mode if only 1 sample, otherwise sample
        is_sample = n_samples > 1

        original_batch_size, context_length, n_features, num_bins = x.shape
        assert n_features == 1, f"BinConv expects n_features=1, got {n_features}"
        
        # Squeeze the features dimension for processing: (batch_size, context_length, num_bins)
        x_squeezed = x.squeeze(2)
        
        # Calculate number of batches needed
        n_batches = (n_samples + max_samples_per_batch - 1) // max_samples_per_batch
        
        all_forecasts = []
        
        for batch_idx in range(n_batches):
            # Calculate actual samples for this batch
            start_sample = batch_idx * max_samples_per_batch
            end_sample = min(start_sample + max_samples_per_batch, n_samples)
            current_n_samples = end_sample - start_sample
            
            # Expand input for current number of samples
            # From (batch_size, context_length, num_bins) to (batch_size * current_n_samples, context_length, num_bins)
            expanded_input = x_squeezed.unsqueeze(1).expand(
                original_batch_size, current_n_samples, context_length, num_bins
            ).reshape(original_batch_size * current_n_samples, context_length, num_bins)
            
            # Initialize current context with expanded input
            current_context = expanded_input.clone()  # (batch_size * current_n_samples, context_length, num_bins)
            forecasts = []
            
            # Generate forecasts autoregressively for each horizon step
            for step in range(horizon):
                # Forward pass to get logits
                logits = self._forward_single(current_context)  # (batch_size * current_n_samples, num_bins)
                
                # Convert logits to probabilities and sample
                probs = torch.sigmoid(logits)  # (batch_size * current_n_samples, num_bins)
                
                # Sample from probabilities to get binary sequences
                pred, _ = get_sequence_from_prob(probs, is_sample=is_sample)  # (batch_size * current_n_samples, num_bins)
                
                # Store forecast for this step
                forecasts.append(pred.unsqueeze(1))  # (batch_size * current_n_samples, 1, num_bins)
                
                # Update context for next step (autoregressive)
                # Remove oldest timestep and add new prediction
                next_input = pred.unsqueeze(1)  # (batch_size * current_n_samples, 1, num_bins)
                current_context = torch.cat([
                    current_context[:, 1:, :],  # Remove first timestep
                    next_input  # Add prediction as new timestep
                ], dim=1)
            
            # Concatenate forecasts along horizon dimension
            batch_forecasts = torch.cat(forecasts, dim=1)  # (batch_size * current_n_samples, horizon, num_bins)
            
            # Reshape back to (batch_size, current_n_samples, horizon, num_bins)
            batch_forecasts = batch_forecasts.view(original_batch_size, current_n_samples, horizon, num_bins)
            
            # Add back the features dimension: (batch_size, current_n_samples, horizon, 1, num_bins)
            batch_forecasts = batch_forecasts.unsqueeze(3)
            
            all_forecasts.append(batch_forecasts)
        
        # Concatenate all sample batches
        final_forecasts = torch.cat(all_forecasts, dim=1)  # (batch_size, n_samples, horizon, 1, num_bins)
        
        return final_forecasts


    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass through the network."""
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

        # Final feed-forward layer (convolutional)
        out = self.conv_layer(
            x,
            self.layers["conv_ffn"],
            None,
            self.kernel_size_ffn,
            is_2d=False
        ).squeeze(1)

        return out
