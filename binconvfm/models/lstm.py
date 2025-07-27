import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
import torch.nn.functional as F
from binconvfm.models.base import BaseForecaster, BaseForecasterModule


class LSTMForecaster(BaseForecaster):
    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 1,
        quantiles: list[float] = [(i + 1) / 10 for i in range(9)],
        batch_size: int = 32,
        num_epochs: int = 10,
        lr: float = 0.001,
        accelerator: str = "cpu",
        enable_progress_bar: bool = True,
        logging: bool = False,
    ):
        """
        Initialize the LSTMForecaster.

        Args:
            input_len (int): Length of input sequence.
            output_len (int): Length of output sequence to forecast.
            hidden_dim (int): Number of hidden units in LSTM.
            n_layers (int): Number of LSTM layers.
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            accelerator (str): Accelerator type ("cpu" or "gpu").
            enable_progress_bar (bool): Whether to show progress bar.
            logging (bool): Enable logging.
        """
        super().__init__(
            quantiles,
            batch_size,
            num_epochs,
            lr,
            accelerator,
            enable_progress_bar,
            logging,
        )
        self.model = LSTMModule(
            hidden_dim,
            n_layers,
            self.quantiles,
            self.lr,
        )


class LSTMModule(BaseForecasterModule):
    def __init__(self, hidden_dim, n_layers, quantiles, lr):
        """
        PyTorch Lightning module for LSTM forecasting.

        Args:
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            lr (float): Learning rate.
        """
        super().__init__(quantiles, lr)
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

    def loss(self, batch: tuple) -> Tensor:
        """
        Compute the mean squared error loss for a batch.

        Args:
            batch (Tensor): Tuple of (input_seq, horizon, n_samples, target_seq).

        Returns:
            Tensor: Loss value.
        """
        input_seq, horizon, n_samples, target_seq = batch
        pred_seq = self.model(
            input_seq, horizon, n_samples=[1], y_teacher=target_seq
        )  # (batch, n_samples, output_len)
        pred_seq = pred_seq.mean(dim=1)
        loss = F.mse_loss(pred_seq, target_seq)
        return loss


class LSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers):
        """
        Initialize the LSTM model for sequence forecasting.

        Args:
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
        """
        super().__init__()

        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, horizon, n_samples, y_teacher=None):
        """
        Vectorized forward pass for the LSTM model with random sampling.

        Args:
            x (Tensor): Input sequence of shape (batch, input_len).
            horizon (int): Forecasting horizon (length of output sequence).
            n_samples (int): Number of samples to generate.
            y_teacher (Tensor, optional): Teacher-forced target sequence (batch, output_len).

        Returns:
            Tensor: Forecast samples of shape (batch, n_samples, output_len).
        """
        batch_size = int(x.size(0))
        device = x.device

        # Prepare encoder input
        x = x.unsqueeze(-1)  # (batch, input_len, 1)
        _, (h, c) = self.lstm(x)  # h, c: (num_layers, batch, hidden_size)

        # Expand hidden and cell states for n_samples
        n_samples = n_samples[0]
        h = h.unsqueeze(2).expand(-1, batch_size, n_samples, -1).contiguous()
        c = c.unsqueeze(2).expand(-1, batch_size, n_samples, -1).contiguous()
        h = h.view(h.size(0), batch_size * n_samples, -1)
        c = c.view(c.size(0), batch_size * n_samples, -1)

        # Initial input: random noise for each sample
        input = torch.rand(batch_size, n_samples, 1, device=device)  # (batch, n_samples, 1)
        input = input.view(batch_size * n_samples, 1, 1)  # (batch * n_samples, 1, 1)

        outputs = []
        horizon = horizon[0]

        for t in range(horizon):
            out, (h, c) = self.lstm(input, (h, c))  # (batch * n_samples, 1, hidden_size)
            pred = self.out_proj(out)  # (batch * n_samples, 1, 1)
            outputs.append(pred.squeeze(-1))  # (batch * n_samples, 1)

            if self.training and y_teacher is not None:
                # Use teacher forcing: repeat y_teacher for n_samples
                next_input = y_teacher[:, t : t + 1]  # (batch, 1)
                next_input = next_input.unsqueeze(1)  # (batch, 1, 1)
                next_input = next_input.expand(batch_size, n_samples, 1)  # (batch, n_samples, 1)
                next_input = next_input.reshape(batch_size * n_samples, 1, 1)  # (batch * n_samples, 1, 1)
                input = next_input
            else:
                input = pred  # use prediction as next input

        outputs = torch.cat(outputs, dim=1)  # (batch * n_samples, horizon)
        outputs = outputs.view(batch_size, n_samples, horizon)  # (batch, n_samples, horizon)
        return outputs
