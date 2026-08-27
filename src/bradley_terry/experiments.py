"""Repeated experiments for parameter-recovery studies."""

import numpy as np

from .estimation import (
    estimate_parameters_with_gradient_descent
)
from .metrics import parameter_recovery_error
from .simulation import (
    generate_uniform_comparison_data
)


def run_parameter_recovery_trials(
    true_parameters,
    number_of_comparisons,
    number_of_runs,
    seed,
    learning_rate=0.1,
    maximum_iterations=1000
):
    """Run repeated simulations and return one error per run."""
    if not isinstance(number_of_runs,(int,np.integer)):
        raise TypeError(
            "number_of_runs must be an integer"
        )

    if number_of_runs<=0:
        raise ValueError(
            "number_of_runs must be positive"
        )

    if not isinstance(seed,(int,np.integer)):
        raise TypeError("seed must be an integer")

    if seed<0:
        raise ValueError("seed must be non-negative")

    parameter_array=np.asarray(
        true_parameters,
        dtype=float
    )

    if parameter_array.ndim!=1:
        raise ValueError(
            "true_parameters must be a one-dimensional array"
        )

    if parameter_array.size<2:
        raise ValueError(
            "at least two true parameters are required"
        )

    if not np.all(np.isfinite(parameter_array)):
        raise ValueError(
            "true_parameters must contain only finite values"
        )

    number_of_items=len(parameter_array)

    recovery_errors=np.empty(
        number_of_runs,
        dtype=float
    )

    independent_seeds=np.random.SeedSequence(
        seed
    ).spawn(number_of_runs)

    for run_index,run_seed in enumerate(independent_seeds):
        run_rng=np.random.default_rng(run_seed)

        winners,losers=generate_uniform_comparison_data(
            parameter_array,
            number_of_comparisons,
            run_rng
        )

        estimated_parameters,_=(
            estimate_parameters_with_gradient_descent(
                number_of_items,
                winners,
                losers,
                learning_rate,
                maximum_iterations
            )
        )

        recovery_errors[run_index]=parameter_recovery_error(
            parameter_array,
            estimated_parameters
        )

    return recovery_errors