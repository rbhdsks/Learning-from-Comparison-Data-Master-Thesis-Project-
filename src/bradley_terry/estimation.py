"""Parameter estimation for the Bradley-Terry model."""

import numpy as np

from .model import (
    average_negative_log_likelihood,
    average_negative_log_likelihood_gradient,
    center_parameters
)


def estimate_parameters_with_gradient_descent(
    number_of_items,
    winners,
    losers,
    learning_rate=0.1,
    maximum_iterations=1000
):
    """Estimate centred parameters using gradient descent."""
    if not isinstance(number_of_items,(int,np.integer)):
        raise TypeError(
            "number_of_items must be an integer"
        )

    if number_of_items<2:
        raise ValueError(
            "number_of_items must be at least two"
        )

    if not np.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError(
            "learning_rate must be a positive finite number"
        )

    if not isinstance(
        maximum_iterations,
        (int,np.integer)
    ):
        raise TypeError(
            "maximum_iterations must be an integer"
        )

    if maximum_iterations<=0:
        raise ValueError(
            "maximum_iterations must be positive"
        )

    estimated_parameters=np.zeros(
        number_of_items,
        dtype=float
    )

    initial_nll=average_negative_log_likelihood(
        estimated_parameters,
        winners,
        losers
    )

    nll_history=[initial_nll]

    for _ in range(maximum_iterations):
        gradient=average_negative_log_likelihood_gradient(
            estimated_parameters,
            winners,
            losers
        )

        estimated_parameters=(
            estimated_parameters
            -learning_rate*gradient
        )

        estimated_parameters=center_parameters(
            estimated_parameters
        )

        current_nll=average_negative_log_likelihood(
            estimated_parameters,
            winners,
            losers
        )

        nll_history.append(current_nll)

    return estimated_parameters,np.asarray(nll_history)