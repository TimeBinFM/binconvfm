"""Lightning DataModule for Monash time series datasets."""

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from typing import Optional

from TSDatasets.monash.get_loaders import get_loaders


class MonashDataModule(LightningDataModule):
    """Lightning DataModule for Monash time series forecasting datasets.
    
    This data module provides train/validation/test data loaders for Monash
    time series datasets in TSF format, with configurable context and prediction lengths.
    
    Args:
        batch_size: Batch size for data loaders.
        horizon: Prediction horizon for test/predict stages.
        input_len: Length of input context window.
        output_len: Length of output prediction for training.
        filename: Path to the TSF file containing the dataset.
        verbose: Whether to print dataset loading information.
    """
    
    def __init__(
        self,
        batch_size: int,
        horizon: int,
        input_len: int,
        output_len: int,
        filename: str = "monash_data/tourism_monthly_dataset.tsf",
        verbose: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.horizon = horizon
        self.input_len = input_len
        self.output_len = output_len
        self.filename = filename
        self.verbose = verbose
        
        # Store datasets
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.pred_ds = None

    def setup(self, stage: str):
        """Set up datasets for the specified stage."""
        if stage == "fit":
            # For training, use output_len for prediction depth
            self.train_ds, self.val_ds = get_loaders(
                context_length=self.input_len,
                prediction_depth_train=1,
                prediction_depth_test=self.output_len,
                filename=self.filename,
                verbose=self.verbose
            )
        elif stage == "test":
            # For testing, use horizon for prediction depth
            _, self.test_ds = get_loaders(
                context_length=self.input_len,
                prediction_depth_train=1,
                prediction_depth_test=self.horizon,
                filename=self.filename,
                verbose=self.verbose
            )
        elif stage == "predict":
            # For prediction, use horizon for prediction depth
            _, self.pred_ds = get_loaders(
                context_length=self.input_len,
                prediction_depth_train=1,
                prediction_depth_test=self.horizon,
                filename=self.filename,
                verbose=self.verbose
            )

    def train_dataloader(self):
        """Return training data loader."""
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return validation data loader."""
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return test data loader."""
        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        """Return prediction data loader."""
        return DataLoader(self.pred_ds, batch_size=self.batch_size)