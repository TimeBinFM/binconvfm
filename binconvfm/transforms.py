from abc import abstractmethod


class BaseTransform:
    """
    Abstract base class for data transformation operations.
    Subclasses should implement fit_transform, transform, and inverse_transform methods.
    """
    @abstractmethod
    def fit_transform(self, data):
        """
        Fit the transformer to the data and transform it.

        Args:
            data (Tensor): Input data to fit and transform.

        Returns:
            Tensor: Transformed data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def transform(self, data):
        """
        Transform the data using the fitted transformer.

        Args:
            data (Tensor): Input data to transform.

        Returns:
            Tensor: Transformed data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def inverse_transform(self, data):
        """
        Inverse transform the data to its original scale.

        Args:
            data (Tensor): Transformed data to inverse transform.

        Returns:
            Tensor: Data in its original scale.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class IdentityTransform(BaseTransform):
    """
    Transformer that returns the data unchanged.
    Useful as a placeholder or for pipelines where no transformation is needed.
    """
    def fit_transform(self, data):
        """
        Return the input data unchanged.

        Args:
            data (Tensor): Input data.

        Returns:
            Tensor: Unchanged input data.
        """
        return data

    def transform(self, data):
        """
        Return the input data unchanged.

        Args:
            data (Tensor): Input data.

        Returns:
            Tensor: Unchanged input data.
        """
        return data

    def inverse_transform(self, data):
        """
        Return the input data unchanged.

        Args:
            data (Tensor): Input data.

        Returns:
            Tensor: Unchanged input data.
        """
        return data


class StandardScaler(BaseTransform):
    """
    Standardize features by removing the mean and scaling to unit variance per sample.
    Stores mean and std for each sample in the batch.
    """
    def __init__(self):
        """
        Initialize the StandardScaler.
        """
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        """
        Fit the scaler on the data and transform it to zero mean and unit variance per sample.

        Args:
            data (Tensor): Input data of shape (batch, ...).

        Returns:
            Tensor: Standardized data.
        """
        self.mean = data.mean(dim=1, keepdim=True)
        self.std = data.std(dim=1, keepdim=True)
        return (data - self.mean) / self.std

    def transform(self, data):
        """
        Transform the data using the previously computed mean and std.

        Args:
            data (Tensor): Input data of shape (batch, ...).

        Returns:
            Tensor: Standardized data.
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inverse the standardization transformation, returning data to its original scale.

        Args:
            data (Tensor): Standardized data of shape (batch, ...).

        Returns:
            Tensor: Data in its original scale.
        """
        std = self.std.unsqueeze(1)
        mean = self.mean.unsqueeze(1)
        return data * std + mean
