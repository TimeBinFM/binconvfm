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
        input_len: int,
        output_len: int,
        hidden_dim: int,
        n_layers: int = 1,
        quantiles: list[float] = [(i + 1) / 10 for i in range(9)],
        n_samples: int = 1000,
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
            n_samples (int): Number of samples for probabilistic output.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            lr (float): Learning rate.
            accelerator (str): Accelerator type ("cpu" or "gpu").
            enable_progress_bar (bool): Whether to show progress bar.
            logging (bool): Enable logging.
        """
        super().__init__(
            input_len,
            output_len,
            quantiles,
            n_samples,
            batch_size,
            num_epochs,
            lr,
            accelerator,
            enable_progress_bar,
            logging,
        )
        self.model = LSTMModule(
            self.input_len,
            self.output_len,
            hidden_dim,
            n_layers,
            self.quantiles,
            self.n_samples,
            self.lr,
        )


class LSTMModule(BaseForecasterModule):
    def __init__(
        self, input_len, output_len, hidden_dim, n_layers, quantiles, n_samples, lr
    ):
        """
        PyTorch Lightning module for LSTM forecasting.

        Args:
            input_len (int): Length of input sequence.
            output_len (int): Length of output sequence.
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
            quantiles (list[float]): List of quantiles for probabilistic forecasting.
            n_samples (int): Number of samples for probabilistic output.
            lr (float): Learning rate.
        """
        super().__init__(input_len, output_len, quantiles, n_samples, lr)
        self.save_hyperparameters()
        self.model = LSTM(input_len, output_len, n_samples, hidden_dim, n_layers)

    def configure_optimizers(self) -> Optimizer:
        """
        Configure and return the optimizer for training.

        Returns:
            Optimizer: The optimizer instance (Adam).
        """
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def loss(self, batch: Tensor) -> Tensor:
        """
        Compute the mean squared error loss for a batch.

        Args:
            batch (Tensor): Tuple of (input_seq, horizon, target_seq).

        Returns:
            Tensor: Loss value.
        """
        input_seq, horizon, target_seq = batch
        pred_seq = self.model(
            input_seq, horizon, target_seq
        )  # (batch, n_samples, output_len)
        pred_seq = pred_seq.mean(dim=1)
        loss = F.mse_loss(pred_seq, target_seq)
        return loss


class LSTM(nn.Module):
    def __init__(self, input_len, output_len, n_samples, hidden_dim, n_layers):
        """
        Initialize the LSTM encoder-decoder model for sequence forecasting.

        Args:
            input_len (int): Length of input sequence.
            output_len (int): Length of output sequence.
            n_samples (int): Number of samples for probabilistic output.
            hidden_dim (int): Hidden dimension of LSTM.
            n_layers (int): Number of LSTM layers.
        """
        super().__init__()
        self.n_samples = n_samples
        self.input_len = input_len
        self.output_len = output_len

        self.encoder = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.LSTM(1, hidden_dim, n_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x, horizon, y_teacher=None):
        """
        Forward pass for the LSTM model.

        Args:
            x (Tensor): Input sequence of shape (batch, input_len).
            horizon (int): Forecasting horizon (length of output sequence).
            y_teacher (Tensor, optional): Teacher-forced target sequence (batch, output_len).

        Returns:
            Tensor: Forecast samples of shape (batch, n_samples, output_len).
        """
        batch_size = x.size(0)
        device = x.device

        # Prepare encoder input
        x = x.unsqueeze(-1)  # (batch, input_len, 1)
        _, (h, c) = self.encoder(x)  # h, c: (num_layers, batch, hidden_size)

        samples = []

        for _ in range(self.n_samples):
            decoder_input = torch.rand(
                batch_size, 1, 1, device=device
            )  # initial input: random
            h_t, c_t = h.clone(), c.clone()
            outputs = []

            for t in range(horizon[0]):
                out, (h_t, c_t) = self.decoder(
                    decoder_input, (h_t, c_t)
                )  # (batch, 1, hidden_size)
                pred = self.out_proj(out)  # (batch, 1, 1)
                outputs.append(pred.squeeze(-1))  # (batch, 1)

                if self.training and y_teacher is not None:
                    decoder_input = y_teacher[:, t : t + 1].unsqueeze(
                        -1
                    )  # (batch, 1, 1)
                else:
                    decoder_input = pred  # use prediction as next input

            outputs = torch.cat(outputs, dim=1)  # (batch, output_len)
            samples.append(outputs.unsqueeze(1))  # (batch, 1, output_len)

        return torch.cat(samples, dim=1)  # (batch, n_samples, output_len)
