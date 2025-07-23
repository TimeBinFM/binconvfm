from torch import nn
import torch
import torch.nn.functional as F
import logging
from typing import Literal, Union
from binconvfm.layers.DynamicTanh import DynamicTanh
from binconvfm.utils.processing import get_preprocessing_pipeline
from binconvfm.utils.forecast import get_sequence_from_prob, most_probable_monotonic_sequence
from binconvfm.utils.reshape import repeat
from pytorch_lightning import LightningModule

# Set up logger
logger = logging.getLogger(__name__)

class BinConv(nn.Module):
    """
    Binary Convolution model for time series forecasting.
    
    This model uses binary quantization combined with 2D and 1D convolutions
    to perform autoregressive forecasting. The architecture processes quantized
    time series data through multiple convolutional blocks with residual connections.
    """
    def __init__(self, context_length: int, is_prob_forecast: bool, num_bins: int, min_bin_value=-10.0,
                 max_bin_value=10.0, kernel_size_across_bins_2d: int = 3,
                 kernel_size_across_bins_1d: int = 3, num_filters_2d: int = 8,
                 num_filters_1d: int = 32, is_cum_sum: bool = False, num_1d_layers: int = 2, num_blocks: int = 3,
                 kernel_size_ffn: int = 51, dropout: float = 0.2, target_dim=1, prediction_length:int = 36,
                 last_layer: Literal["conv", "fc"] = 'conv',
                 scaler_type: Union[Literal["standard", "temporal", "None"], None] = None) -> None:
        super().__init__()
        assert kernel_size_across_bins_2d % 2 == 1, "2D kernel size must be odd"
        assert kernel_size_across_bins_1d % 2 == 1, "1D kernel size must be odd"

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_bins = num_bins
        self.target_dim = target_dim
        logger.info(f'BinConv initialized with dropout: {dropout}, target_dim: {self.target_dim}')
        self.is_prob_forecast = is_prob_forecast
        self.num_filters_2d = num_filters_2d
        self.num_filters_1d = num_filters_1d
        self.kernel_size_across_bins_2d = kernel_size_across_bins_2d
        self.kernel_size_across_bins_1d = kernel_size_across_bins_1d
        self.is_cum_sum = is_cum_sum

        logger.info(f'Per sample scaler type: {scaler_type}')
        # Initialize preprocessing pipelines using the new utils.processing module
        if scaler_type is None:
            self.preprocessors = None
            logger.debug('No preprocessors initialized')
        elif scaler_type == 'standard':
            self.preprocessors = [
                get_preprocessing_pipeline(
                    scaler_type='standard',
                    quantizer_type='binary',
                    num_bins=num_bins,
                    min_val=min_bin_value,
                    max_val=max_bin_value
                ) for _ in range(self.target_dim)
            ]
            logger.info(f'Initialized {self.target_dim} standard preprocessing pipelines with binary quantization')
        elif scaler_type == 'temporal':
            self.preprocessors = [
                get_preprocessing_pipeline(
                    scaler_type='mean',  # Use mean scaler as temporal equivalent
                    quantizer_type='binary',
                    num_bins=num_bins,
                    min_val=min_bin_value,
                    max_val=max_bin_value
                ) for _ in range(self.target_dim)
            ]
            logger.info(f'Initialized {self.target_dim} temporal preprocessing pipelines with binary quantization')
        else:
            raise ValueError(f"The scaler type {scaler_type} is not supported. Use 'standard', 'temporal', or None.")
        
        # Track if preprocessors are fitted
        self._preprocessors_fitted = False
        self.num_1d_layers = num_1d_layers
        self.num_blocks = num_blocks
        self.kernel_size_ffn = kernel_size_ffn
        self.dropout = nn.Dropout(dropout)
        # Conv2d over (context_length, num_bins)

        assert num_filters_2d == num_filters_1d, "todo: change the self.act shape if not"
        self.act = nn.ModuleList([
            nn.ModuleList([
                DynamicTanh(normalized_shape=num_filters_2d if i < self.num_1d_layers else context_length,
                            channels_last=False)
                for i in range(1)  # applied only after conv2d
                # for i in range(self.num_1d_layers + 1)  # applied only after conv2d
            ]) for _ in range(self.num_blocks)
        ])

        logger.debug(f'Activation functions after conv2d: {len(self.act)} blocks with {len(self.act[0])} layers each')

        layers = []
        for i in range(target_dim):
            conv2d = nn.ModuleList([nn.Conv2d(
                in_channels=1,
                out_channels=self.num_filters_2d,
                # kernel_size=(context_length if i == 0 else kernel_size_across_bins_2d, kernel_size_across_bins_2d),
                kernel_size=(context_length, kernel_size_across_bins_2d),
                bias=True
            ) for _ in range(num_blocks)
            ])
            conv1d = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv1d(in_channels=num_filters_2d if i == 0 else num_filters_1d,
                              out_channels=context_length if i == num_1d_layers - 1 else num_filters_1d,
                              kernel_size=kernel_size_across_bins_1d, bias=True,
                              groups=num_filters_1d)
                    for i in range(num_1d_layers)
                ]) for _ in range(num_blocks)
            ])
            self.last_layer = last_layer
            if last_layer == 'conv':
                conv_ffn = nn.Conv1d(
                    in_channels=context_length,
                    out_channels=1,
                    kernel_size=kernel_size_ffn,  # large kernel size?
                    groups=1,
                    bias=True
                )
            elif last_layer == 'fc':
                class MeanOverChannel(nn.Module):
                    def __init__(self, dim=-2):
                        super().__init__()
                        self.dim = dim

                    def forward(self, x):
                        return x.mean(dim=self.dim)


                conv_ffn = nn.Sequential(
                    MeanOverChannel(dim=-2),  # Averages over the channel dimension
                    nn.Linear(in_features=self.num_bins, out_features=self.num_bins, bias=True)
                ) #was used for mlp ablation study
            else:
                assert False, f"last layer {last_layer} is not supported"
            assert num_filters_2d == num_filters_1d, "todo: change the self.act shape if not"
            act = nn.ModuleList([
                nn.ModuleList([
                    # DynamicTanh(normalized_shape=num_filters_2d if i == 0 else num_filters_1d, channels_last=False)
                    DynamicTanh(normalized_shape=num_filters_2d if i < self.num_1d_layers else context_length,
                                channels_last=False)
                    for i in range(1)
                    # applied after conv2d, and all conv1d including the last one
                ]) for _ in range(self.num_blocks)
            ])
            logger.debug(f'Initialized layers for target dimension {i}:')
            logger.debug(f'  - Conv2D layers: {len(conv2d)} blocks')
            logger.debug(f'  - Conv1D layers: {len(conv1d)} blocks with {len(conv1d[0])} layers each')
            logger.debug(f'  - FFN layer type: {last_layer}')
            layers.append(nn.ModuleDict({
                'conv2d': conv2d,
                'conv1d': conv1d,
                'conv_ffn': conv_ffn,
                'act': act,
            }))
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def _pad_channels(tensor: torch.Tensor, pad_size: int, pad_val_left=1.0, pad_val_right=0.0):
        if pad_size == 0:
            return tensor
        left = torch.full((*tensor.shape[:-1], pad_size), pad_val_left, device=tensor.device)
        right = torch.full((*tensor.shape[:-1], pad_size), pad_val_right, device=tensor.device)
        return torch.cat([left, tensor, right], dim=-1)

    def conv_layer(self, x: torch.Tensor, conv_func, act_func, kernel_size: int, is_2d: bool, ):
        # kernel_size = self.kernel_size_across_bins_2d if is_2d else self.kernel_size_across_bins_1d
        pad = kernel_size // 2 if kernel_size > 1 else 0
        x_padded = self._pad_channels(x, pad)
        if is_2d:
            x_padded = x_padded.unsqueeze(1)
        conv_out = conv_func(x_padded)  # (batch_size, num_filters_2d, num_bins)

        if is_2d:
            conv_out = conv_out.squeeze(2)
        if act_func is not None:
            conv_out = act_func(conv_out)
        return conv_out

    def forward(self, x, layer_id=0):
        if len(x.shape) == 4:
            x = x.squeeze()
        x = x.float()
        # x: (batch_size, context_length, num_bins)
        batch_size, context_length, num_bins = x.shape
        assert context_length == self.context_length, "Mismatch in context length"

        for j in range(self.num_blocks):
            residual = x
            x = self.conv_layer(x, self.layers[layer_id]["conv2d"][j], self.layers[layer_id]["act"][j][0],
                                self.kernel_size_across_bins_2d, True)
            for i in range(self.num_1d_layers):
                x = self.conv_layer(x, self.layers[layer_id]["conv1d"][j][i], F.relu,
                                    self.kernel_size_across_bins_1d, False)
            x = self.dropout(x)
            x = x + residual
        if self.last_layer == 'conv':
            out = self.conv_layer(x, self.layers[layer_id]["conv_ffn"], None, self.kernel_size_ffn, False).squeeze(1)
        else:
            out = self.layers[layer_id]["conv_ffn"]
        if self.is_cum_sum:
            raise NotImplementedError("Cumulative sum is disabled as it degrades performance")
            # out = torch.flip(torch.cumsum(torch.flip(out, dims=[1]), dim=1), dims=[1])
        return out

    @torch.inference_mode()
    def forecast(self, batch_data, num_samples=None):
        do_sample = num_samples is not None and num_samples > 1 and self.is_prob_forecast
        inputs = batch_data.unsqueeze(2)
        logger.debug(f'Forecast input shape: {inputs.shape}')
        forecasts_list = []
        for c in range(inputs.shape[2]):
            if self.preprocessors is not None:
                self.preprocessors[c].fit(inputs[:, :, c:c + 1].reshape(-1))
                c_inputs = self.preprocessors[c].transform(inputs[:, :, c:c + 1])
            else:
                c_inputs = inputs[:, :, c:c + 1]

            if do_sample:
                c_inputs = repeat(c_inputs.unsqueeze(1), num_samples, 1)  # (B, NS, T, D)
                batch_size = c_inputs.shape[0]
                c_inputs = c_inputs.view(-1, *c_inputs.shape[2:])
            current_context = c_inputs.clone()
            c_forecasts = []
            for _ in range(self.prediction_length):
                pred = F.sigmoid(self(current_context))  # (B, D)
                # logger.debug(f'Prediction range: [{pred.min():.4f}, {pred.max():.4f}]')
                pred, _ = get_sequence_from_prob(pred, do_sample)
                pred = pred.int()
                c_forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
                next_input = pred.unsqueeze(1)

                if len(current_context.shape) == 4:
                    next_input = next_input.unsqueeze(1)
                current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

            c_forecasts = torch.cat(c_forecasts, dim=1)
            if self.preprocessors is not None:
                c_forecasts = self.preprocessors[c].inverse_transform(c_forecasts)
            if do_sample:
                c_forecasts = c_forecasts.view(batch_size, num_samples, *c_forecasts.shape[1:])
            else:
                c_forecasts = c_forecasts.unsqueeze(1)  # (B, 1,  T, D)
            if inputs.shape[2] > 1:
                c_forecasts = c_forecasts.unsqueeze(-2)  # (B, 1, T, D, num_bins)
            forecasts_list.append(c_forecasts)
        forecasts = torch.concat(forecasts_list, dim=-2)  # was 2
        return forecasts


    def predict(self, x: torch.Tensor, prediction_length: int) -> torch.Tensor:
        """
        Autoregressive prediction over `prediction_length` steps.

        Args:
            x: Input tensor of shape (B, context_length, num_bins)
            prediction_length: number of future steps to forecast

        Returns:
            Tensor of shape (B, prediction_length, num_bins)
        """
        device = next(self.parameters()).device
        x = x.to(device)
        current_context = x.clone()
        forecasts = []
        for i in range(prediction_length):
            pred = F.sigmoid(self(current_context))  # (B, D)
            # pred = (pred >= 0.5).int()
            if i % 10 == 0:  # Log every 10th step to avoid spam
                logger.debug(f'Prediction step {i}/{prediction_length}')
            pred, _ = most_probable_monotonic_sequence(pred)
            pred = pred.int()
            forecasts.append(pred.unsqueeze(1))  # (B, 1, D)
            next_input = pred.unsqueeze(1)
            current_context = torch.cat([current_context[:, 1:], next_input], dim=1)

        return torch.cat(forecasts, dim=1)  # (B, T, D)


class LightningBinConv(BinConv, LightningModule):
    """
    PyTorch Lightning wrapper for BinConv model.
    
    Provides training loop, loss computation, and optimizer configuration
    for the binary convolution forecasting model. Handles preprocessing
    pipeline fitting automatically during setup.
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
        logger.info(f'LightningBinConv initialized with learning rate: {lr}')

    def setup(self, stage: str = None):
        """
        Setup method called by Lightning.
        
        Args:
            stage: Training stage ('fit', 'validate', 'test', 'predict')
        """
        logger.info(f'Lightning setup called for stage: {stage}')
        logger.info('BinConv uses per-sample preprocessing (foundation model approach)')

    def training_step(self, batch, batch_idx):
        """
        Training step for one batch using vectorized per-sample preprocessing.
        
        Args:
            batch: Tuple of (inputs, targets) tensors where:
                   - inputs: Raw input data (batch_size, context_length, target_dim)
                   - targets: Target data that will be preprocessed per-sample
            batch_idx: Index of the current batch
            
        Returns:
            torch.Tensor: Training loss for this batch
        """
        inputs, targets = batch
        batch_size = inputs.shape[0]
        
        # Process inputs using per-sample preprocessing (but vectorized)
        if self.preprocessors is not None:
            # Process the first dimension (target_dim=0) for all samples in batch
            processed_inputs = self._process_batch(inputs[:, :, 0:1], 0)  # (batch_size, context_length, num_bins)
        else:
            processed_inputs = inputs
        
        # Forward pass through the model
        logits = []
        for b in range(batch_size):
            sample_input = processed_inputs[b]  # (context_length, num_bins) or (context_length, target_dim)
            sample_logits = self(sample_input.unsqueeze(0))  # Add batch dim for model
            logits.append(sample_logits.squeeze(0))  # Remove batch dim
        
        logits = torch.stack(logits)  # (batch_size, num_bins)
        
        # Process targets using per-sample preprocessing
        if self.preprocessors is not None:
            # Process targets (first timestep only for next-step prediction)
            processed_targets = self._process_batch(targets[:, 0:1, 0:1], 0)  # (batch_size, 1, num_bins)
            target_binary = processed_targets.squeeze(1)  # (batch_size, num_bins)
        else:
            target_binary = targets.squeeze(1).squeeze(1)  # Assume already processed
        
        # Compute loss for entire batch
        loss = F.binary_cross_entropy_with_logits(logits, target_binary)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx % 100 == 0:  # Log every 100 batches to avoid spam
            logger.debug(f'Batch {batch_idx}: Training loss = {loss.item():.6f}')
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for one batch using vectorized per-sample preprocessing.
        
        Args:
            batch: Tuple of (inputs, targets) tensors
            batch_idx: Index of the current batch
            
        Returns:
            torch.Tensor: Validation loss for this batch
        """
        inputs, targets = batch
        batch_size = inputs.shape[0]
        
        # Process inputs using per-sample preprocessing (but vectorized)
        if self.preprocessors is not None:
            # Process the first dimension (target_dim=0) for all samples in batch
            processed_inputs = self._process_batch(inputs[:, :, 0:1], 0)  # (batch_size, context_length, num_bins)
        else:
            processed_inputs = inputs
        
        # Forward pass through the model
        logits = []
        for b in range(batch_size):
            sample_input = processed_inputs[b]  # (context_length, num_bins) or (context_length, target_dim)
            sample_logits = self(sample_input.unsqueeze(0))  # Add batch dim for model
            logits.append(sample_logits.squeeze(0))  # Remove batch dim
        
        logits = torch.stack(logits)  # (batch_size, num_bins)
        
        # Process targets using per-sample preprocessing
        if self.preprocessors is not None:
            # Process targets (first timestep only for next-step prediction)
            processed_targets = self._process_batch(targets[:, 0:1, 0:1], 0)  # (batch_size, 1, num_bins)
            target_binary = processed_targets.squeeze(1)  # (batch_size, num_bins)
        else:
            target_binary = targets.squeeze(1).squeeze(1)  # Assume already processed
        
        # Compute loss for entire batch
        loss = F.binary_cross_entropy_with_logits(logits, target_binary)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def _process_batch(self, batch_data, dim_idx):
        """
        Process a batch of samples using per-sample preprocessing (vectorized).
        
        Each sample in the batch is preprocessed independently using its own statistics,
        maintaining the foundation model approach while being efficient.
        
        Args:
            batch_data: Batch tensor of shape (batch_size, seq_len, features)
            dim_idx: Dimension index (for multi-dimensional targets)
            
        Returns:
            torch.Tensor: Processed batch with quantization applied
        """
        if self.preprocessors is None or dim_idx >= len(self.preprocessors):
            return batch_data
            
        batch_size, seq_len, features = batch_data.shape
        
        # Get the preprocessing pipeline for this dimension
        preprocessor = self.preprocessors[dim_idx]
        
        # Process each sample independently but in vectorized manner
        processed_samples = []
        
        for b in range(batch_size):
            sample = batch_data[b:b+1]  # (1, seq_len, features)
            
            # Flatten the sample for preprocessing
            sample_flat = sample.reshape(-1)  # (seq_len * features,)
            
            # Apply preprocessing pipeline (scaler + quantizer)
            processed_flat = preprocessor.fit_transform(sample_flat)
            
            # Reshape back: if quantized, will have shape (seq_len * features, num_bins)
            if processed_flat.dim() > 1:
                # Quantized output - reshape to (1, seq_len, num_bins)
                processed_sample = processed_flat.view(1, seq_len, -1)
            else:
                # Scaled output - reshape to original shape
                processed_sample = processed_flat.view(1, seq_len, features)
            
            processed_samples.append(processed_sample)
        
        # Stack all processed samples back into batch
        return torch.cat(processed_samples, dim=0)

    def configure_optimizers(self):
        """
        Configure optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Adam optimizer with specified learning rate
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        logger.info(f'Configured Adam optimizer with lr={self.lr}')
        return optimizer