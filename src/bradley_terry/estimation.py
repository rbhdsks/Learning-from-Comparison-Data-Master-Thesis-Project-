"""Parameter estimation for the Bradley-Terry model."""

import numpy as np

from .model import (
    average_negative_log_likelihood,
    average_negative_log_likelihood_gradient,
    center_parameters,
)


def _validate_estimation_settings(
    number_of_items,
    learning_rate,
    maximum_iterations,
):
    """Validate settings shared by gradient-descent estimators."""
    if not isinstance(
        number_of_items,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_items must be an integer"
        )

    if number_of_items < 2:
        raise ValueError(
            "number_of_items must be at least two"
        )

    if (
        not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be a positive finite number"
        )

    if not isinstance(
        maximum_iterations,
        (int, np.integer),
    ):
        raise TypeError(
            "maximum_iterations must be an integer"
        )

    if maximum_iterations <= 0:
        raise ValueError(
            "maximum_iterations must be positive"
        )


def estimate_parameters_with_gradient_descent_trace(
    number_of_items,
    comparisons,
    learning_rate=0.1,
    maximum_iterations=1_000,
):
    """Estimate parameters and record their complete trajectory."""
    _validate_estimation_settings(
        number_of_items,
        learning_rate,
        maximum_iterations,
    )

    estimated_parameters = np.zeros(
        number_of_items,
        dtype=float,
    )

    parameter_history = np.empty(
        (
            maximum_iterations + 1,
            number_of_items,
        ),
        dtype=float,
    )

    nll_history = np.empty(
        maximum_iterations + 1,
        dtype=float,
    )

    parameter_history[0] = (
        estimated_parameters
    )

    nll_history[0] = (
        average_negative_log_likelihood(
            estimated_parameters,
            comparisons,
        )
    )

    for iteration in range(
        1,
        maximum_iterations + 1,
    ):
        gradient = (
            average_negative_log_likelihood_gradient(
                estimated_parameters,
                comparisons,
            )
        )

        estimated_parameters = (
            estimated_parameters
            - learning_rate * gradient
        )

        estimated_parameters = center_parameters(
            estimated_parameters
        )

        parameter_history[iteration] = (
            estimated_parameters
        )

        nll_history[iteration] = (
            average_negative_log_likelihood(
                estimated_parameters,
                comparisons,
            )
        )

    return (
        estimated_parameters,
        nll_history,
        parameter_history,
    )


def estimate_parameters_with_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.1,
    maximum_iterations=1_000,
):
    """Estimate centred parameters using gradient descent."""
    (
        estimated_parameters,
        nll_history,
        _,
    ) = estimate_parameters_with_gradient_descent_trace(
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        maximum_iterations=maximum_iterations,
    )

    return (
        estimated_parameters,
        nll_history,
    )