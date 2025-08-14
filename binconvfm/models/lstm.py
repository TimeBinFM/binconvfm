import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
from binconvfm.models.base import BaseForecaster, BaseLightningModule, BaseTorchModule
from typing import List


class LSTMForecaster(BaseForecaster):
    def __init__(
        self,
        # Common parameters - explicit
        horizon: int = 1,
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
        # LSTM-specific parameters - go to kwargs
        **model_kwargs
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
        lstm_defaults = {
            "hidden_dim": 64,
            "n_layers": 1,
        }

        # Update defaults with provided kwargs
        for key, default_value in lstm_defaults.items():
            if key not in model_kwargs:
                model_kwargs[key] = default_value

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
            **model_kwargs  # Pass LSTM-specific parameters
        )

    def _create_model(self):
        """
        Create and assign the LSTMModule model to this forecaster.
        """
        self.model = LSTMModule(
            horizon=self.horizon,
            n_samples=self.n_samples,
            quantiles=self.quantiles,
            lr=self.lr,
            transform=self.transform,
            hidden_dim=self.model_kwargs["hidden_dim"],
            n_layers=self.model_kwargs["n_layers"],
        )


class LSTMModule(BaseLightningModule):
    def __init__(
        self,
        horizon: int,
        n_samples: int,
        quantiles: List[float],
        lr: float,
        transform: List[str],
        hidden_dim: int,
        n_layers: int,
    ):
        """
        Initialize the LSTMModule for PyTorch Lightning training.

        Args:
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
            horizon (int): Forecasting horizon.
            n_samples (int): Number of samples to generate.
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            lr (float): Learning rate.
        """
        super().__init__(horizon, n_samples, quantiles, lr, transform)
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
        self, input_seq: torch.Tensor, target_seq: torch.Tensor, batch_idx: int
    ) -> torch.Tensor:
        """
        Compute the mean squared error (MSE) loss between the predicted and target sequences for a batch.

        Args:
            input_seq (Tensor): Input sequence tensor for the model.
            target_seq (Tensor): Ground truth target sequence tensor.
            batch_idx (int): Index of the current batch.
        Returns:
            Tensor: Computed MSE loss value for the batch.
        """
        pred_seq = self.model(
            input_seq, self.horizon, n_samples=1, y=target_seq
        )  # (batch, n_samples, output_len)
        pred_seq = pred_seq.mean(dim=1)
        loss = F.mse_loss(pred_seq, target_seq)
        return loss


class LSTM(BaseTorchModule):
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

    def _forward(
        self, x: torch.Tensor, horizon: int, n_samples: int, y: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Vectorized forward pass for the LSTM model with random sampling.

        Args:
            x (Tensor): Input sequence of shape (batch, input_len, dim=1).
            horizon (int): Forecasting horizon (length of output sequence).
            n_samples (int): Number of samples to generate.
            y (Tensor, optional): Teacher-forced target sequence (batch, output_len, dim=1).

        Returns:
            Tensor: Forecast samples of shape (batch, n_samples, horizon, dim=1).
        """
        assert (
            x.shape[2] == 1
        ), "Input sequence must have a single feature dimension (dim=1)"
        assert (
            y.shape[2] == 1 if y is not None else True
        ), "Target sequence must have a single feature dimension (dim=1)"

        batch_size = x.size(0)
        device = x.device

        # Encode input sequence
        _, (h, c) = self.lstm(x)  # h, c: (num_layers, batch, hidden_size)

        # Expand hidden and cell states for n_samples
        h = h.unsqueeze(2).expand(-1, batch_size, n_samples, -1).contiguous()
        c = c.unsqueeze(2).expand(-1, batch_size, n_samples, -1).contiguous()
        h = h.view(h.size(0), batch_size * n_samples, -1)
        c = c.view(c.size(0), batch_size * n_samples, -1)

        # Initial input: random noise for each sample
        input = torch.rand(
            batch_size, n_samples, 1, device=device
        )  # (batch, n_samples, 1)
        input = input.view(batch_size * n_samples, 1, 1)  # (batch * n_samples, 1, 1)

        outputs = []

        for t in range(horizon):
            out, (h, c) = self.lstm(
                input, (h, c)
            )  # (batch * n_samples, 1, hidden_size)
            pred = self.out_proj(out)  # (batch * n_samples, 1, 1)
            outputs.append(pred)  # (batch * n_samples, 1, 1)

            if self.training and y is not None:
                # Use teacher forcing: repeat y for n_samples
                next_input = y[:, t : t + 1, :]  # (batch, 1, 1)
                next_input = next_input.expand(
                    batch_size, n_samples, 1
                )  # (batch, n_samples, 1)
                next_input = next_input.reshape(
                    batch_size * n_samples, 1, 1
                )  # (batch * n_samples, 1, 1)
                input = next_input
            else:
                input = pred  # use prediction as next input

        outputs = torch.cat(outputs, dim=1)  # (batch * n_samples, horizon, 1)
        outputs = outputs.view(
            batch_size, n_samples, horizon, 1
        )  # (batch, n_samples, horizon, 1)
        return outputs
