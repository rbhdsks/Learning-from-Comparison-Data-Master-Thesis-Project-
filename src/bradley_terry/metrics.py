"""Evaluation metrics for parameter recovery."""

import numpy as np

from .model import center_parameters


def parameter_recovery_error(
    true_parameters,
    estimated_parameters
):
    """Calculate RMSE after centring both parameter vectors."""
    true_parameter_array=np.asarray(
        true_parameters,
        dtype=float
    )

    estimated_parameter_array=np.asarray(
        estimated_parameters,
        dtype=float
    )

    if true_parameter_array.ndim!=1:
        raise ValueError(
            "true_parameters must be a one-dimensional array"
        )

    if estimated_parameter_array.ndim!=1:
        raise ValueError(
            "estimated_parameters must be a one-dimensional array"
        )

    if true_parameter_array.shape!=estimated_parameter_array.shape:
        raise ValueError(
            "true and estimated parameters must have the same length"
        )

    if not np.all(np.isfinite(true_parameter_array)):
        raise ValueError(
            "true_parameters must contain only finite values"
        )

    if not np.all(np.isfinite(estimated_parameter_array)):
        raise ValueError(
            "estimated_parameters must contain only finite values"
        )

    centered_true_parameters=center_parameters(
        true_parameter_array
    )

    centered_estimated_parameters=center_parameters(
        estimated_parameter_array
    )

    parameter_differences=(
        centered_estimated_parameters
        -centered_true_parameters
    )

    return float(
        np.sqrt(
            np.mean(parameter_differences**2)
        )
    )