from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import LightningModule
import torch.nn as nn
from binconvfm.utils.metrics import mase, crps
from abc import abstractmethod
from typing import List
from binconvfm.transform.factory import TransformFactory


class BaseForecaster:
    def __init__(
        self,
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
        **kwargs
    ):
        """
        Initializes the base model with the specified configuration.

        Args:
            horizon (int, optional): Forecast horizon. Defaults to 1.
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
        self.horizon = horizon
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
        self.kwargs = kwargs  # Store model-specific parameters
        self.trainer = None
        self.model = None

    @abstractmethod
    def _create_model(self):
        pass

    def set_horizon(self, horizon: int) -> None:
        self.horizon = horizon
        if self.model is not None:
            self.model.horizon = horizon

    def set_n_samples(self, n_samples: int) -> None:
        self.n_samples = n_samples
        if self.n_samples is not None:
            self.model.n_samples = n_samples

    def _create_trainer(self) -> None:
        self.trainer = Trainer(
            enable_progress_bar=self.enable_progress_bar,
            max_epochs=self.num_epochs,
            log_every_n_steps=self.log_every_n_steps,
            logger=self.logger,
            accelerator=self.accelerator,
        )

    def fit(self, train_dataloader, val_dataloader=None) -> None:
        self._create_trainer()
        self._create_model()
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    def evaluate(self, test_dataloader):
        if self.trainer is None:
            self._create_trainer()
        if self.model is None:
            self._create_model()
        metrics = self.trainer.test(self.model, test_dataloader)
        return metrics[0]  # Assuming single test dataloader and single result

    def predict(self, pred_dataloader):
        if self.trainer is None:
            self._create_trainer()
        if self.model is None:
            self._create_model()
        return self.trainer.predict(self.model, pred_dataloader)


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        horizon: int,
        n_samples: int,
        quantiles: List[float],
        lr: float,
        transform: List[str],
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
        self.horizon = horizon
        self.n_samples = n_samples
        self.quantiles = quantiles
        self.lr = lr
        # Always create a pipeline for consistency
        self.transform = TransformFactory.create_pipeline(transform)

    def training_step(self, batch, batch_idx: int):
        input_seq, target_seq = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Transform target sequence using the same params
        target_seq = self.transform.transform(target_seq, transform_params)
        loss = self.loss(input_seq, target_seq, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        input_seq, target_seq = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Transform target sequence using the same params
        target_seq = self.transform.transform(target_seq, transform_params)
        loss = self.loss(input_seq, target_seq, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        input_seq, target_seq = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Generate predictions
        pred_seq = self.model(input_seq, self.horizon, self.n_samples)
        # Inverse transform predictions to original scale
        pred_seq = self.transform.inverse_transform(pred_seq, transform_params)
        metrics = {
            "mase": mase(pred_seq, target_seq),
            "crps": crps(pred_seq, target_seq, self.quantiles),
        }
        self.log_dict(metrics, prog_bar=True)

    def predict_step(self, batch, batch_idx: int):
        input_seq, _ = batch
        # Fit transform on input sequence and store params
        input_seq, transform_params = self.transform.fit_transform(input_seq)
        # Generate predictions
        pred_seq = self.model(input_seq, self.horizon, self.n_samples)
        # Inverse transform predictions to original scale
        pred_seq = self.transform.inverse_transform(pred_seq, transform_params)
        return pred_seq

    @abstractmethod
    def loss(self, input_seq, target_seq, batch_idx: int):
        """
        Compute the loss for a batch.

        Args:
            input_seq (Tensor): The input sequence tensor for the batch.
            target_seq (Tensor): The target sequence tensor for the batch.
            batch_idx (int): Index of the batch.

        Returns:
            Tensor: Computed loss value for the batch.
        """
        raise NotImplementedError("Subclasses must implement this method")


# TODO: let's delete it from the base interface; not all models will follow this pattern
class BaseTorchModule(nn.Module):
    def forward(self, x, horizon, n_samples, y=None):
        """
        Vectorized forward pass for the base torch model with random sampling.

        Args:
            x (Tensor): Input sequence of shape (batch, input_len, dim).
            horizon (int): Forecasting horizon (length of output sequence).
            n_samples (int): Number of samples to generate.
            y (Tensor, optional): Teacher-forced target sequence (batch, output_len, dim).

        Returns:
            Tensor: Forecast samples of shape (batch, n_samples, output_len, dim).
        """
        assert x.dim() == 3, "Input must be a 3D tensor (batch, input_len, dim)"
        assert (
            y is None or y.dim() == 3
        ), "Target must be a 3D tensor (batch, output_len, dim) if provided"
        return self._forward(x, horizon, n_samples, y)

    @abstractmethod
    def _forward(self, x, horizon, n_samples, y=None):
        """
        Abstract method to be implemented by subclasses for the forward pass.

        Args:
            x (Tensor): Input sequence.
            horizon (int): Forecasting horizon.
            n_samples (int): Number of samples to generate.
            y (Tensor, optional): Teacher-forced target sequence.

        Returns:
            Tensor: Output of the model.
        """
        raise NotImplementedError("Subclasses must implement this method")
