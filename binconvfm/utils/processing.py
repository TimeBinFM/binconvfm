import torch
import logging
from typing import Literal
# Set up logger
logger = logging.getLogger(__name__)


class TorchOneHotQuantizer:
    """
    Pure PyTorch implementation of one-hot quantizer.
    
    Transforms continuous values into one-hot encoded vectors based on fixed bins.
    Each value is mapped to the closest bin and represented as a one-hot vector.
    """
    
    def __init__(self, num_bins=1000, min_val=-10.0, max_val=10.0):
        """
        Initialize the one-hot quantizer.
        
        Args:
            num_bins (int): Number of quantization bins
            min_val (float): Minimum value for quantization range
            max_val (float): Maximum value for quantization range
        """
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        
        logger.info(f'OneHotQuantizer initialized - num_bins: {num_bins}, min_val: {self.min_val}, max_val: {self.max_val}')
        
        self.bin_edges_ = torch.linspace(self.min_val, self.max_val, self.num_bins + 1)
        self.bin_values_ = 0.5 * (self.bin_edges_[:-1] + self.bin_edges_[1:])

    def fit(self, values, y=None):
        """
        Fit the quantizer (no-op for compatibility with sklearn interface).
        
        Args:
            values: Input values (unused for fixed binning)
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the quantizer and transform the data in one step.
        
        Args:
            X: Input tensor to transform
            y: Target values (unused, for sklearn compatibility)
            **fit_params: Additional fitting parameters (unused)
            
        Returns:
            torch.Tensor: One-hot encoded tensor
        """
        return self.transform(X)

    def transform(self, values):
        """
        Transform values to one-hot representation.
        
        Each value is assigned to the nearest bin and converted to a one-hot vector
        where only the corresponding bin position is 1.
        
        Args:
            values: Input tensor of any shape
            
        Returns:
            torch.Tensor: One-hot tensor with shape (*values.shape, num_bins)
        """
        # Move bin_edges to same device as input
        bin_edges = self.bin_edges_.to(values.device)
        
        bin_indices = torch.bucketize(values, bin_edges, right=False) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)

        one_hot = torch.zeros(*values.shape, self.num_bins, device=values.device, dtype=values.dtype)
        one_hot.scatter_(-1, bin_indices.unsqueeze(-1), 1.0)
        return one_hot

    def inverse_transform(self, one_hot_values):
        """
        Convert one-hot representation back to values.
        
        Takes the argmax of each one-hot vector and maps it back to the
        corresponding bin center value.
        
        Args:
            one_hot_values: One-hot tensor with last dimension = num_bins
            
        Returns:
            torch.Tensor: Reconstructed values using bin center values
        """
        bin_values = self.bin_values_.to(one_hot_values.device)
        indices = one_hot_values.argmax(dim=-1)
        return bin_values[indices]


class TorchBinaryQuantizer:
    """
    Pure PyTorch implementation of binary quantizer.
    
    Transforms continuous values into binary vectors where each element indicates
    whether the input value exceeds the corresponding bin threshold.
    """
    
    def __init__(self, num_bins=1000, min_val=-10.0, max_val=10.0):
        """
        Initialize the binary quantizer.
        
        Args:
            num_bins (int): Number of quantization bins
            min_val (float): Minimum value for quantization range
            max_val (float): Maximum value for quantization range
        """
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        
        logger.info(f'BinaryQuantizer initialized - num_bins: {num_bins}, min_val: {min_val}, max_val: {max_val}')

        self.bin_edges_ = torch.linspace(self.min_val, self.max_val, self.num_bins + 1)
        self.bin_values_ = 0.5 * (self.bin_edges_[:-1] + self.bin_edges_[1:])

    def fit(self, values, y=None):
        """
        Fit the quantizer (no-op for compatibility with sklearn interface).
        
        Args:
            values: Input values (unused for fixed binning)
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            self: Returns self for method chaining
        """
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the quantizer and transform the data in one step.
        
        Args:
            X: Input tensor to transform
            y: Target values (unused, for sklearn compatibility)
            **fit_params: Additional fitting parameters (unused)
            
        Returns:
            torch.Tensor: Binary encoded tensor
        """
        return self.transform(X)

    def transform(self, values):
        """
        Convert each value into a binary vector based on the bin thresholds.
        
        For each input value, creates a binary vector where element i is 1 if
        the value is greater than or equal to the i-th bin threshold, 0 otherwise.
        
        Args:
            values: Input tensor of any shape
            
        Returns:
            torch.Tensor: Binary tensor with shape (*values.shape, num_bins)
        """
        # Move bin_edges to same device as input
        bin_edges = self.bin_edges_.to(values.device)
        
        # Add dimension for broadcasting
        values_expanded = values.unsqueeze(-1)  # Shape: (..., 1)
        
        # Create thresholds for broadcasting
        bin_thresholds = bin_edges[1:]  # Shape: (num_bins,)
        
        # Broadcast comparison
        binary_vectors = (values_expanded >= bin_thresholds).float()
        
        return binary_vectors

    def inverse_transform(self, binary_vectors):
        """
        Reconstruct values from binary quantized representation.
        
        For each binary vector, finds the highest active bin (last 1 in the sequence)
        and returns the midpoint of that bin. If all bins are zero, returns the
        first bin value as fallback.
        
        Args:
            binary_vectors: Binary tensor with last dimension = num_bins
            
        Returns:
            torch.Tensor: Reconstructed values using bin center values
        """
        bin_values = self.bin_values_.to(binary_vectors.device)
        
        # Find the last active bin (highest bin that's active)
        reversed_bin = torch.flip(binary_vectors, dims=[-1])
        idx_first_one_reversed = torch.argmax(reversed_bin, dim=-1)
        idx_last_one = self.num_bins - 1 - idx_first_one_reversed
        
        reconstructed = bin_values[idx_last_one]

        # Handle edge case: all zeros
        all_zero_mask = binary_vectors.sum(dim=-1) == 0
        reconstructed = torch.where(all_zero_mask, bin_values[0], reconstructed)

        return reconstructed


class TorchStandardScaler:
    """
    Pure PyTorch implementation of standard scaler (z-score normalization).
    
    Standardizes features by removing the mean and scaling to unit variance.
    The standard score of a sample x is calculated as: z = (x - mean) / std
    """
    
    def __init__(self):
        """
        Initialize the standard scaler.
        
        The scaler computes mean and standard deviation during fitting.
        """
        self.mean_ = None
        self.std_ = None
        self.fitted = False
        logger.debug('TorchStandardScaler initialized')

    def fit(self, X, y=None):
        """
        Fit the scaler by computing the mean and standard deviation.
        
        Args:
            X: Input tensor of any shape
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            self: Returns self for method chaining
        """
        self.mean_ = torch.mean(X)
        self.std_ = torch.std(X, unbiased=False)
        
        # Avoid division by zero
        if self.std_ == 0:
            self.std_ = torch.tensor(1.0, device=X.device, dtype=X.dtype)
            logger.warning('Standard deviation is zero, setting to 1.0 to avoid division by zero')
        
        self.fitted = True
        logger.debug(f'TorchStandardScaler fitted with mean: {self.mean_}, std: {self.std_}')
        return self

    def transform(self, X):
        """
        Transform the data by standardizing (z-score normalization).
        
        Args:
            X: Input tensor to standardize
            
        Returns:
            torch.Tensor: Standardized tensor with mean=0 and std=1
            
        Raises:
            ValueError: If the scaler has not been fitted yet
        """
        if not self.fitted:
            raise ValueError("This scaler has not been fitted yet.")
        
        # Move parameters to same device as input
        mean = self.mean_.to(X.device) if isinstance(self.mean_, torch.Tensor) else self.mean_
        std = self.std_.to(X.device) if isinstance(self.std_, torch.Tensor) else self.std_
        return (X - mean) / std

    def fit_transform(self, X, y=None):
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            X: Input tensor to fit and transform
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            torch.Tensor: Standardized tensor
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Inverse transform the standardized data back to original scale.
        
        Args:
            X_scaled: Standardized tensor to inverse transform
            
        Returns:
            torch.Tensor: Data restored to original scale
            
        Raises:
            ValueError: If the scaler has not been fitted yet
        """
        if not self.fitted:
            raise ValueError("This scaler has not been fitted yet.")
        
        # Move parameters to same device as input
        mean = self.mean_.to(X_scaled.device) if isinstance(self.mean_, torch.Tensor) else self.mean_
        std = self.std_.to(X_scaled.device) if isinstance(self.std_, torch.Tensor) else self.std_
        return X_scaled * std + mean


class TorchTimeSeriesMeanScaler:
    """
    Pure PyTorch implementation of time series mean scaler.
    
    Scales input data by dividing by the global mean of the training data.
    This is useful for normalizing time series data while preserving relative patterns.
    """
    
    def __init__(self):
        """
        Initialize the mean scaler.
        
        The scaler computes a global mean during fitting and uses it for scaling.
        """
        self.means_ = None
        self.fitted = False
        logger.debug('TimeSeriesMeanScaler initialized')

    def fit(self, X, y=None):
        """
        Fit the scaler by computing the global mean of the input data.
        
        Args:
            X: Input tensor of any shape
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            self: Returns self for method chaining
        """
        self.means_ = torch.mean(X)
        # Avoid division by zero
        if self.means_ == 0:
            self.means_ = torch.tensor(1.0, device=X.device, dtype=X.dtype)
            logger.warning('Mean is zero, setting to 1.0 to avoid division by zero')
        
        self.fitted = True
        logger.debug(f'TimeSeriesMeanScaler fitted with mean: {self.means_}')
        return self

    def transform(self, X):
        """
        Transform the data by dividing by the fitted mean.
        
        Args:
            X: Input tensor to scale
            
        Returns:
            torch.Tensor: Scaled tensor where each element is divided by the global mean
            
        Raises:
            ValueError: If the scaler has not been fitted yet
        """
        if not self.fitted:
            raise ValueError("This scaler has not been fitted yet.")
        
        # Move means to same device as input
        means = self.means_.to(X.device) if isinstance(self.means_, torch.Tensor) else self.means_
        return X / means

    def fit_transform(self, X, y=None):
        """
        Fit the scaler and transform the data in one step.
        
        Args:
            X: Input tensor to fit and transform
            y: Target values (unused, for sklearn compatibility)
            
        Returns:
            torch.Tensor: Scaled tensor
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Inverse transform the scaled data back to original scale.
        
        Multiplies the scaled data by the fitted mean to recover the original scale.
        
        Args:
            X_scaled: Scaled tensor to inverse transform
            
        Returns:
            torch.Tensor: Data restored to original scale
            
        Raises:
            ValueError: If the scaler has not been fitted yet
        """
        if not self.fitted:
            raise ValueError("This scaler has not been fitted yet.")
        
        # Move means to same device as input
        means = self.means_.to(X_scaled.device) if isinstance(self.means_, torch.Tensor) else self.means_
        return X_scaled * means


class TorchPipeline:
    """
    Pure PyTorch implementation of sklearn-like Pipeline.
    
    Chains multiple transformers together, applying them sequentially.
    Each transformer's output becomes the input to the next transformer.
    """
    
    def __init__(self, steps):
        """
        Initialize the pipeline with a sequence of transformers.
        
        Args:
            steps: list of (name, transformer) tuples where each transformer
                  implements fit, transform, and inverse_transform methods
        """
        self.steps = {}
        self.step_names = []
        
        for name, transformer in steps:
            self.steps[name] = transformer
            self.step_names.append(name)
        
        logger.info(f'TorchPipeline initialized with steps: {self.step_names}')

    def fit(self, X, y=None):
        """
        Fit all transformers in the pipeline sequentially.
        
        Each transformer is fitted on the output of the previous transformer.
        
        Args:
            X: Input data to fit the pipeline on
            y: Target values (passed to each transformer's fit method)
            
        Returns:
            self: Returns self for method chaining
        """
        current_data = X
        for name in self.step_names:
            transformer = self.steps[name]
            logger.debug(f'Fitting transformer: {name}')
            transformer.fit(current_data, y)
            current_data = transformer.transform(current_data)
        
        logger.info('Pipeline fitting completed')
        return self

    def transform(self, X):
        """
        Apply all transformations in the pipeline sequentially.
        
        Args:
            X: Input data to transform
            
        Returns:
            torch.Tensor: Transformed data after applying all pipeline steps
        """
        current_data = X
        for name in self.step_names:
            transformer = self.steps[name]
            current_data = transformer.transform(current_data)
        return current_data

    def fit_transform(self, X, y=None):
        """
        Fit all transformers and transform the data in one step.
        
        Args:
            X: Input data to fit and transform
            y: Target values (passed to each transformer's fit method)
            
        Returns:
            torch.Tensor: Transformed data
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Apply inverse transformations in reverse order.
        
        Undoes the effect of the pipeline by applying each transformer's
        inverse_transform method in reverse order.
        
        Args:
            X: Transformed data to inverse transform
            
        Returns:
            torch.Tensor: Data restored to original form
        """
        current_data = X
        for name in reversed(self.step_names):
            transformer = self.steps[name]
            current_data = transformer.inverse_transform(current_data)
        return current_data


def get_preprocessing_pipeline(
    scaler_type: Literal['standard', 'mean'] = 'mean',
    quantizer_type: Literal['binary', 'onehot'] = 'binary',
    num_bins: int = 1000,
    min_val: float = -15,
    max_val: float = 15
) -> TorchPipeline:
    """
    Create a preprocessing pipeline using pure PyTorch components.
    
    The pipeline first scales the data, then applies quantization based on the
    specified types. This flexible interface allows mixing different scalers
    with different quantizers.
    
    Args:
        scaler_type (Literal['standard', 'mean']): Type of scaler to use
            - 'standard': StandardScaler (z-score normalization)
            - 'mean': TimeSeriesMeanScaler (divide by global mean)
        quantizer_type (Literal['binary', 'onehot']): Type of quantizer to use
            - 'binary': Binary quantization (threshold-based encoding)
            - 'onehot': One-hot quantization (categorical encoding)
        num_bins (int): Number of quantization bins for the quantizer
        min_val (float): Minimum value for quantization range
        max_val (float): Maximum value for quantization range
        
    Returns:
        TorchPipeline: Complete preprocessing pipeline ready for fitting and transformation
        
    Raises:
        ValueError: If invalid scaler_type or quantizer_type is provided
    """
    # Select scaler
    if scaler_type == 'standard':
        scaler = TorchStandardScaler()
    elif scaler_type == 'mean':
        scaler = TorchTimeSeriesMeanScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}. Must be 'standard' or 'mean'")
    
    # Select quantizer
    if quantizer_type == 'binary':
        quantizer = TorchBinaryQuantizer(num_bins=num_bins, min_val=min_val, max_val=max_val)
    elif quantizer_type == 'onehot':
        quantizer = TorchOneHotQuantizer(num_bins=num_bins, min_val=min_val, max_val=max_val)
    else:
        raise ValueError(f"Unknown quantizer_type: {quantizer_type}. Must be 'binary' or 'onehot'")
    
    pipeline = TorchPipeline([
        ('scaler', scaler),
        ('quantizer', quantizer)
    ])
    
    logger.info(f'Created preprocessing pipeline with {scaler_type} scaler and {quantizer_type} quantizer')
    return pipeline

