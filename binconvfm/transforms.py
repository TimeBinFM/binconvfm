class BaseTransform:
    def fit_transform(self, data):
        """
        Fit the transformer to the data and transform it.

        Args:
            data (Tensor): Input data to fit and transform.

        Returns:
            Tensor: Transformed data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def transform(self, data):
        """
        Transform the data using the fitted transformer.

        Args:
            data (Tensor): Input data to transform.

        Returns:
            Tensor: Transformed data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

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
    def fit_transform(self, data):
        return data

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class StandardScaler(BaseTransform):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        self.mean = data.mean(dim=1, keepdim=True)
        self.std = data.std(dim=1, keepdim=True)
        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        std = self.std.unsqueeze(1)
        mean = self.mean.unsqueeze(1)
        return data * std + mean
