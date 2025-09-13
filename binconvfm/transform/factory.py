import inspect
from typing import List, Union, Dict, Any, Tuple
from . import scalers, quantizers
from .pipeline import Pipeline


class TransformFactory:
    """Factory class for creating data transformation pipelines from string specifications."""

    @classmethod
    def create_pipeline(cls, transform_list: List[str], transform_args: Dict[str, Dict[str, Any]] = None) -> Pipeline:
        """
        Create a transform pipeline from a list of transform names with optional parameters.

        Args:
            transform_list: List of transform names (in CamelCase)
            transform_args: Dict mapping transform names to their parameters
                          e.g., {"BinaryQuantizer": {"num_bins": 100, "min_val": -5.0}}

        Returns:
            Pipeline object with the specified transforms

        Raises:
            ValueError: If an unknown transform is provided
        """
        transform_args = transform_args or {}
        steps = []
        
        for transform_name in transform_list:
            transform_class = cls._find_transform_class(transform_name)
            
            # Get parameters for this transform
            transform_params = transform_args.get(transform_name, {})

            # Create transform instance with parameters
            transform_instance = transform_class(**transform_params)

            # Add to pipeline
            steps.append((transform_name, transform_instance))

        return Pipeline(steps)

    @classmethod
    def _find_transform_class(cls, transform_name: str):
        """
        Find transform class by exact name match.

        Args:
            transform_name: Exact class name of the transform

        Returns:
            Transform class

        Raises:
            ValueError: If transform class not found
        """
        # Search in scalers module
        if hasattr(scalers, transform_name):
            transform_class = getattr(scalers, transform_name)
            if inspect.isclass(transform_class):
                return transform_class

        # Search in quantizers module
        if hasattr(quantizers, transform_name):
            transform_class = getattr(quantizers, transform_name)
            if inspect.isclass(transform_class):
                return transform_class

        # If not found, list available transforms
        available = cls.get_available_transforms()
        raise ValueError(
            f"Unknown transform '{transform_name}'. Available transforms: {available}"
        )

    @classmethod
    def get_available_transforms(cls) -> List[str]:
        """Get list of available transform class names."""
        available = []

        # Check scalers module
        for name, obj in inspect.getmembers(scalers, inspect.isclass):
            if hasattr(obj, "fit_transform") or hasattr(obj, "transform"):
                available.append(name)

        # Check quantizers module
        for name, obj in inspect.getmembers(quantizers, inspect.isclass):
            if hasattr(obj, "fit_transform") or hasattr(obj, "transform"):
                available.append(name)

        return available
