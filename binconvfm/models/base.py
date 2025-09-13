from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import LightningModule
from binconvfm.utils.metrics import mase, crps, nmae
from abc import abstractmethod, ABC
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from binconvfm.transform.factory import TransformFactory


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for neural network models used in time series forecasting.
    
    This interface defines the common API that all forecasting models must implement.
    Models inheriting from this class should implement the forward and sample methods.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for sequence forecasting.
        
        Args:
            x (Tensor): Input sequence of shape (batch, seq_len, features).
            y (Tensor): Target sequence for teacher forcing, shape (batch, horizon, features).
            
        Returns:
            Tensor: Forecast of shape (batch, horizon, features).
        """
        pass

    @abstractmethod
    def forecast(self, x: torch.Tensor, horizon: int, n_samples: int) -> torch.Tensor:
        """
        Sample from the model's output distribution.
        
        Args:
            x (Tensor): Input sequence of shape (batch, seq_len, features).
            horizon (int): Forecasting horizon (length of output sequence).
            n_samples (int): Number of samples to generate.
            
        Returns:
            Tensor: Sampled sequences of shape (batch, n_samples, horizon, features).
        """
        pass


class BaseForecaster:
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
        Initializes the base model with the specified configuration.

        Args:
            n_samples (int, optional): Number of samples for probabilistic forecasting. Defaults to 1000.
            quantiles (list[float], optional): List of quantiles to predict. Defaults to [(i + 1) / 10 for i in range(9)].
            batch_size (int, optional): Batch size for training. Defaults to 32.
            num_epochs (int, optional): Number of training epochs. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 0.001.
            accelerator (str, optional): Device to use for training ('cpu', 'gpu', etc.). Defaults to "cpu".
            enable_progress_bar (bool, optional): Whether to display a progress bar during training. Defaults to True.
            logging (bool, optional): Whether to enable logging. Defaults to False.
            log_every_n_steps (int, optional): Frequency of logging steps. Defaults to 10.
            transform (List[str], optional): List of data transformations to apply. Defaults to ["identity"].
        """
        self.n_samples = n_samples
        self.quantiles = quantiles
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.accelerator = accelerator
        self.enable_progress_bar = enable_progress_bar
        self.logger = False
        if logging:
            self.logger = CSVLogger(save_dir="logs")
        self.log_every_n_steps = log_every_n_steps
        self.transform = transform
        self.transform_args = transform_args or {}
        self.kwargs = kwargs  # Store model-specific parameters
        self.trainer = None
        self.model = None
        self.horizon = None

    @abstractmethod
    def _create_model(self):
        """Create and assign the forecasting model. To be implemented by subclasses."""
        pass

    def _create_trainer(self) -> None:
        """Create the PyTorch Lightning Trainer instance."""
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=self.log_every_n_steps,
            logger=self.logger,
            accelerator=self.accelerator,
        )

    def fit(self, datamodule) -> None:
        """Train the model using the provided dataloaders."""
        self._create_trainer()
        self._create_model()
        self.trainer.fit(
            model=self.model,
            datamodule=datamodule,
        )

    def evaluate(self, datamodule) -> dict:
        """Evaluate the model on the test dataloader."""
        if self.trainer is None:
            self._create_trainer()
        if self.model is None:
            self._create_model()
        metrics = self.trainer.test(self.model, datamodule=datamodule)
        return metrics[0]  # Assuming single test dataloader and single result

    def predict(self, datamodule, horizon):
        """Generate predictions for the given dataloader and horizon."""
        if self.trainer is None:
            self._create_trainer()
        if self.model is None:
            self._create_model()
        self.model.horizon = horizon  # Set the horizon for the model
        return self.trainer.predict(self.model, datamodule=datamodule)

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save the model weights to the specified filepath using PyTorch Lightning Trainer's checkpointing.
        Args:
            filepath (str): Path to save the checkpoint file.
        """
        if self.model is None:
            self._create_model()
        if self.trainer is None:
            self._create_trainer()
        self.trainer.save_checkpoint(filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load the model weights from the specified checkpoint file using PyTorch Lightning.
        Args:
            filepath (str): Path to the checkpoint file.
        """
        if self.model is None:
            self._create_model()
        self.model = self.model.__class__.load_from_checkpoint(filepath)


class BaseLightningModule(LightningModule):
    def __init__(
            self,
            n_samples: int,
            quantiles: List[float],
            lr: float,
            transform: List[str],
            transform_args: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initializes the forecasting model with specified parameters.

            horizon (int): The forecasting horizon, i.e., number of future time steps to predict.
            n_samples (int): Number of samples to generate for probabilistic forecasting.
            quantiles (list or tuple): List of quantiles to predict (e.g., [0.1, 0.5, 0.9]).
            lr (float): Learning rate for the optimizer.
            transform (List[str]): List of data transformations to apply.

        Raises:
            ValueError: If an unknown transform is provided.
        """
        super().__init__()
        self.save_hyperparameters()
        self.n_samples = n_samples
        self.quantiles = quantiles
        self.lr = lr
        # Always create a pipeline for consistency
        self.transform_args = transform_args or {}
        self.transform = TransformFactory.create_pipeline(transform, self.transform_args)
        self.horizon = None

    def training_step(self, batch, batch_idx: int):
        """Run a single training step and return loss."""
        input_seq, target_seq = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Transform target sequence using the same params
        target_seq = self.transform.transform(target_seq, transform_params)
        output_seq = self.model(input_seq, target_seq)
        loss = self.loss(output_seq, target_seq, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """Run a single validation step and return loss."""
        input_seq, target_seq = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Transform target sequence using the same params
        target_seq = self.transform.transform(target_seq, transform_params)
        output_seq = self.model(input_seq, target_seq)
        loss = self.loss(output_seq, target_seq, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        """Run a single test step and log metrics."""
        input_seq, target_seq = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Generate predictions
        pred_seq = self.model.forecast(input_seq, target_seq.shape[1], self.n_samples)
        # Inverse transform predictions to original scale
        pred_seq = self.transform.inverse_transform(pred_seq, transform_params)
        metrics = {
            "mase": mase(pred_seq, target_seq),
            "crps": crps(pred_seq, target_seq, self.quantiles),
            "nmae": nmae(pred_seq, target_seq),
        }
        self.log_dict(metrics, prog_bar=True)

    def predict_step(self, batch, batch_idx: int):
        """Run a single prediction step and return predictions."""
        input_seq, _ = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Generate predictions
        pred_seq = self.model.forecast(input_seq, self.horizon, self.n_samples)
        # Inverse transform predictions to original scale
        pred_seq = self.transform.inverse_transform(pred_seq, transform_params)
        return pred_seq

    @abstractmethod
    def loss(self, output_seq, target_seq, batch_idx: int):
        """Compute and return the loss for a batch."""
        raise NotImplementedError("Subclasses must implement this method")
