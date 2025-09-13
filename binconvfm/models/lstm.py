import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
from binconvfm.models.base import BaseForecaster, BaseLightningModule, BaseModel
from typing import List, Optional, Dict, Any


class LSTMForecaster(BaseForecaster):
    def __init__(
            self,
            n_samples: int = 1000,
            quantiles: list[float] = [(i + 1) / 10 for i in range(9)],
            batch_size: int = 32,
            num_epochs: int = 10,
            lr: float = 0.001,
            accelerator: str = "cpu",
            enable_progress_bar: bool = True,
            logging: bool = False,
            log_every_n_steps: int = 10,
            transform: List[str] = ["IdentityTransform"],
            transform_args: Optional[Dict[str, Dict[str, Any]]] = None,
            **kwargs
    ):
        """
        Initialize the LSTMForecaster.

        Args:
            hidden_dim (int): Number of hidden units in LSTM.
            n_layers (int): Number of LSTM layers.
            horizon (int): Length of output sequence to forecast.
            n_samples (int): Number of samples to generate.
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            accelerator (str): Accelerator type ("cpu" or "gpu").
            enable_progress_bar (bool): Whether to show progress bar.
            logging (bool): Enable logging.
        """

        # Set default values for LSTM-specific parameters if not provided
        defaults = {
            "hidden_dim": 64,
            "n_layers": 1,
        }
        kwargs = {**defaults, **kwargs}

        super().__init__(
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
            **kwargs,  # Pass LSTM-specific parameters
        )

    def _create_model(self):
        """
        Create and assign the LSTMModule model to this forecaster.
        """
        self.model = LSTMModule(
            n_samples=self.n_samples,
            quantiles=self.quantiles,
            lr=self.lr,
            transform=self.transform,
            transform_args=self.transform_args,
            **self.kwargs,
        )


class LSTMModule(BaseLightningModule):
    def __init__(
            self,
            hidden_dim: int,
            n_layers: int,
            n_samples: int,
            quantiles: List[float],
            lr: float,
            transform: List[str],
            transform_args: Optional[Dict[str, Dict[str, Any]]] = None,
            horizon: Optional[int] = None,
    ):
        """
        Initialize the LSTMModule for PyTorch Lightning training.

        Args:
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
            n_samples (int): Number of samples to generate.
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            lr (float): Learning rate.
        """
        super().__init__(n_samples, quantiles, lr, transform, transform_args, horizon)
        self.save_hyperparameters()
        self.model = LSTM(hidden_dim, n_layers)

    def configure_optimizers(self) -> Optimizer:
        """
        Configure and return the optimizer for training.

        Returns:
            Optimizer: The optimizer instance (Adam).
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def loss(
            self, output_seq: torch.Tensor, target_seq: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Compute the mean squared error (MSE) loss between the predicted and target sequences for a batch.

        Args:
            output_seq (Tensor): Output sequence tensor from the model.
            target_seq (Tensor): Ground truth target sequence tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: Computed MSE loss value for the batch.
        """
        loss = F.mse_loss(output_seq, target_seq)
        return loss


class LSTM(BaseModel):
    def __init__(self, hidden_dim: int = 64, n_layers: int = 1):
        """
        Initialize the LSTM model for sequence forecasting.

        Args:
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for sequence forecasting with optional teacher forcing.

        Args:
            x (Tensor): Input sequence of shape (batch, seq_len, 1).
            y (Tensor): Target sequence for teacher forcing, shape (batch, horizon, 1).

        Returns:
            Tensor: Forecast of shape (batch, horizon, 1) if y is provided, else (batch, 1).
        """
        assert (
                x.shape[2] == 1
        ), "Input sequence must have a single feature dimension (dim=1)"

        # Teacher forcing for multi-step prediction
        horizon = y.size(1)
        outputs = []
        out, (h, c) = self.lstm(x)
        last_hidden = out[:, -1, :]
        input_next = self.out_proj(last_hidden).unsqueeze(1)  # (batch, 1, 1)
        outputs.append(input_next)

        for t in range(1, horizon):
            # Use ground truth as next input (teacher forcing)
            input_tf = y[:, t - 1, :].unsqueeze(1)  # (batch, 1, 1)
            out, (h, c) = self.lstm(input_tf, (h, c))
            last_hidden = out[:, -1, :]
            out_proj = self.out_proj(last_hidden).unsqueeze(1)
            outputs.append(out_proj)

        outputs = torch.cat(outputs, dim=1)  # (batch, horizon, 1)
        return outputs

    def forecast(self, x: torch.Tensor, horizon: int, n_samples: int) -> torch.Tensor:
        """
        Sample from the model's output distribution in parallel using the batch dimension,
        with stochasticity introduced by random initialization of the hidden state.

        Args:
            x (Tensor): Input sequence of shape (batch, seq_len, 1).
            horizon (int): Forecasting horizon (length of output sequence).
            n_samples (int): Number of samples to generate.

        Returns:
            Tensor: Sampled sequence of shape (batch, n_samples, horizon, 1).
        """
        batch_size, seq_len, feat_dim = x.shape
        device = x.device
        hidden_dim = self.lstm.hidden_size
        n_layers = self.lstm.num_layers

        # Duplicate input batch n_samples times along batch dimension
        x_rep = x.unsqueeze(1).repeat(
            1, n_samples, 1, 1
        )  # (batch, n_samples, seq_len, 1)
        x_rep = x_rep.view(
            batch_size * n_samples, seq_len, feat_dim
        )  # (batch * n_samples, seq_len, 1)

        # Random initialization of hidden and cell states for stochasticity
        h0 = torch.randn(n_layers, batch_size * n_samples, hidden_dim, device=device)
        c0 = torch.randn(n_layers, batch_size * n_samples, hidden_dim, device=device)

        preds = []
        out, (h, c) = self.lstm(x_rep, (h0, c0))
        last_hidden = out[:, -1, :]
        out_proj = self.out_proj(last_hidden)
        preds.append(out_proj.unsqueeze(1))  # (batch * n_samples, 1, 1)

        for _ in range(1, horizon):
            input_next = out_proj.unsqueeze(1)  # (batch * n_samples, 1, 1)
            out, (h, c) = self.lstm(input_next, (h, c))
            last_hidden = out[:, -1, :]
            out_proj = self.out_proj(last_hidden)
            preds.append(out_proj.unsqueeze(1))  # (batch * n_samples, 1, 1)

        preds = torch.cat(preds, dim=1)  # (batch * n_samples, horizon, 1)
        # Reshape to (batch, n_samples, horizon, 1)
        preds = preds.view(batch_size, n_samples, horizon, 1)
        return preds
