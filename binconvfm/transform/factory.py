import inspect
from typing import List
from . import scalers, quantizers
from .pipeline import Pipeline


class TransformFactory:
    """Factory class for creating data transformation pipelines from string specifications."""

    @classmethod
    def create_pipeline(cls, transform_list: List[str]) -> Pipeline:
        """
        Create a transform pipeline from a list of transform names.

        Args:
            transform_list: List of transform names (in CamelCase)

        Returns:
            Pipeline object with the specified transforms

        Raises:
            ValueError: If an unknown transform is provided
        """
        steps = []
        for transform_name in transform_list:
            transform_class = cls._find_transform_class(transform_name)

            # Create transform instance
            transform_instance = transform_class()

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
