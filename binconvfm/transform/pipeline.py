"""
Pipeline for chaining multiple stateless transformations.

This module provides a sklearn-like Pipeline class for torch tensors that chains
multiple stateless transformations while maintaining parameter storage for each
transformation step to enable proper inverse transformation.

The Pipeline class maintains the stateless nature of individual transforms by
storing parameters returned from each step and passing them through the chain.
"""

import torch
import logging
from typing import List, Tuple, Dict, Union, Any

from .base import BaseTransform

# Set up logger
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Stateless pipeline for chaining multiple transformations.

    This class provides a sklearn-like interface for chaining multiple
    transformation steps while maintaining the stateless nature of each
    individual transformation. Parameters are managed at the pipeline level
    and passed through the transformation chain.

    The pipeline maintains parameter storage for each transformation step
    to enable proper inverse transformation while keeping the pipeline itself
    stateless (parameters are returned rather than stored).

    Example:
        # Create pipeline with multiple steps
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('quantizer', BinaryQuantizer(num_bins=1000))
        ])

        # Transform data
        transformed, params = pipeline.transform(data)

        # Inverse transform
        original = pipeline.inverse_transform(transformed, params)
    """

    def __init__(self, steps: List[Tuple[str, BaseTransform]]):
        """
        Initialize the pipeline.

        Args:
            steps: List of (name, transformer) tuples defining the pipeline steps.
                   Each transformer must implement the BaseTransform interface.
        """
        if not isinstance(steps, list):
            raise TypeError("steps must be a list of (name, transformer) tuples")

        if not steps:
            logger.warning("Creating empty pipeline with no transformation steps")

        for i, step in enumerate(steps):
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValueError(f"Step {i} must be a tuple of (name, transformer)")

            name, transformer = step
            if not isinstance(name, str):
                raise ValueError(f"Step {i} name must be a string, got {type(name)}")

            if not isinstance(transformer, BaseTransform):
                raise ValueError(
                    f"Step {i} transformer must be a BaseTransform, "
                    f"got {type(transformer)}"
                )

        self.steps = steps
        self.step_names = [name for name, _ in steps]
        self.transformers = [transformer for _, transformer in steps]

        logger.debug(f"Pipeline initialized with steps: {self.step_names}")

    def __len__(self) -> int:
        """Return number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, idx: Union[int, str]) -> BaseTransform:
        """
        Get transformer by index or name.

        Args:
            idx: Step index (int) or step name (str)

        Returns:
            The transformer at the specified step
        """
        if isinstance(idx, int):
            return self.transformers[idx]
        elif isinstance(idx, str):
            try:
                step_idx = self.step_names.index(idx)
                return self.transformers[step_idx]
            except ValueError:
                raise KeyError(f"Pipeline has no step named '{idx}'")
        else:
            raise TypeError(f"Pipeline indices must be int or str, got {type(idx)}")

    def fit(self, data: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Fit all transformers and return their parameters.

        This method applies each transformation in sequence, computing parameters
        for each step using the output of the previous step. This ensures that
        each transformer sees the data in the same form it will during transform().

        Args:
            data: Input tensor to fit the pipeline on

        Returns:
            List of parameter dictionaries, one for each transformation step
        """
        current_data = data
        all_params = []

        for i, (name, transformer) in enumerate(self.steps):
            logger.debug(f"Fitting transformation step {i+1}/{len(self.steps)}: {name}")

            # Fit the transformer and get parameters
            step_params = transformer.fit(current_data)
            all_params.append(step_params)

            # Transform data for next step
            current_data = transformer.transform(current_data, step_params)

        return all_params

    def transform(
        self, data: torch.Tensor, params: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Apply all transformations in sequence.

        Args:
            data: Input tensor to transform
            params: Optional pre-computed parameters for all steps. If provided,
                   uses these instead of computing new ones via fit().
                   Should be a list of parameter dicts, one for each step.

        Returns:
            final_transformed_data
        """
        if params is not None and len(params) != len(self.steps):
            raise ValueError(
                f"Number of parameter sets ({len(params)}) must match "
                f"number of pipeline steps ({len(self.steps)})"
            )

        current_data = data

        for i, (name, transformer) in enumerate(self.steps):
            logger.debug(
                f"Applying transformation step {i+1}/{len(self.steps)}: {name}"
            )

            # Use provided params if available, otherwise None (compute new ones)
            step_params = params[i] if params is not None else None

            current_data = transformer.transform(current_data, step_params)

        return current_data

    def inverse_transform(
        self, data: torch.Tensor, params: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Apply inverse transformations in reverse order.

        Args:
            data: Transformed tensor to inverse transform
            params: List of parameters from transform() method, one for each step

        Returns:
            Tuple of (original_scale_tensor, list_of_parameters_used)
        """
        if len(params) != len(self.steps):
            raise ValueError(
                f"Number of parameter sets ({len(params)}) must match "
                f"number of pipeline steps ({len(self.steps)})"
            )

        current_data = data

        # Apply inverse transformations in reverse order
        for i, ((name, transformer), step_params) in enumerate(
            zip(reversed(self.steps), reversed(params))
        ):
            step_num = len(self.steps) - i
            logger.debug(
                f"Applying inverse transformation step {step_num}/{len(self.steps)}: {name}"
            )
            current_data = transformer.inverse_transform(current_data, step_params)

        return current_data

    def fit_transform(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Fit all transformers and apply transformations.

        This is a convenience method equivalent to calling fit() and then
        transform() with the computed parameters.

        Args:
            data: Input tensor to fit and transform

        Returns:
            Tuple of (final_transformed_data, list_of_parameters_per_step)
        """
        params = self.fit(data)
        transformed_data = self.transform(data, params)
        return transformed_data, params

    def get_step_names(self) -> List[str]:
        """
        Get names of all pipeline steps.

        Returns:
            List of step names
        """
        return self.step_names.copy()

    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters for all transformers in the pipeline.

        Returns:
            Dictionary mapping step names to transformer parameters
        """
        params = {}
        for name, transformer in self.steps:
            # Get transformer's initialization parameters if available
            if hasattr(transformer, "get_params"):
                params[name] = transformer.get_params()
            else:
                # Fallback: include basic info about the transformer
                params[name] = {
                    "class": transformer.__class__.__name__,
                    "input_dims": getattr(transformer, "input_dims", None),
                    "inverse_input_dims": getattr(
                        transformer, "inverse_input_dims", None
                    ),
                }

        return params
