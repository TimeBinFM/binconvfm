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
