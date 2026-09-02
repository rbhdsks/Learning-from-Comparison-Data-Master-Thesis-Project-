"""Handcrafted optimization algorithms for the Bradley-Terry model."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .estimation import (
    estimate_parameters_with_gradient_descent_trace,
)
from .model import (
    average_negative_log_likelihood,
    average_negative_log_likelihood_gradient,
    center_parameters,
)


@dataclass
class OptimizationResult:
    """Store the output and history of one optimization run."""

    optimizer_name: str
    parameters: np.ndarray
    nll_history: np.ndarray
    parameter_history: np.ndarray
    update_count_history: np.ndarray
    runtime_seconds: float


def _validate_optimization_inputs(
    number_of_items,
    comparisons,
    learning_rate,
    number_of_epochs,
):
    """Validate settings shared by all optimizers."""

    if not isinstance(number_of_items, (int, np.integer)):
        raise TypeError("number_of_items must be an integer")

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least two")

    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError(
            "learning_rate must be a positive finite number"
        )

    if not isinstance(number_of_epochs, (int, np.integer)):
        raise TypeError("number_of_epochs must be an integer")

    if number_of_epochs <= 0:
        raise ValueError("number_of_epochs must be positive")

    comparison_array = np.asarray(comparisons)

    if (
        comparison_array.ndim != 2
        or comparison_array.shape[1] != 2
    ):
        raise ValueError(
            "comparisons must contain (winner, loser) pairs"
        )

    if len(comparison_array) == 0:
        raise ValueError("at least one comparison is required")

    if not np.issubdtype(
        comparison_array.dtype,
        np.integer,
    ):
        raise TypeError(
            "comparisons must contain integer item indices"
        )

    comparison_array = comparison_array.astype(
        int,
        copy=False,
    )

    if np.any(comparison_array < 0):
        raise ValueError(
            "comparison indices cannot be negative"
        )

    if np.any(comparison_array >= number_of_items):
        raise ValueError(
            "comparison indices exceed number_of_items"
        )

    if np.any(
        comparison_array[:, 0]
        == comparison_array[:, 1]
    ):
        raise ValueError(
            "an item cannot be compared with itself"
        )

    return comparison_array


def _initialise_optimization_history(
    number_of_items,
    comparisons,
):
    """Create zero parameters and record their initial state."""

    parameters = np.zeros(
        number_of_items,
        dtype=float,
    )

    initial_nll = average_negative_log_likelihood(
        parameters,
        comparisons,
    )

    nll_history = [initial_nll]
    parameter_history = [parameters.copy()]
    update_count_history = [0]

    return (
        parameters,
        nll_history,
        parameter_history,
        update_count_history,
    )


def _create_epoch_batches(
    number_of_observations,
    batch_size,
    random_generator,
    shuffle,
):
    """Create observation-index batches for one epoch."""

    observation_indices = np.arange(
        number_of_observations
    )

    if shuffle:
        observation_indices = random_generator.permutation(
            observation_indices
        )

    batches = []

    for start_index in range(
        0,
        number_of_observations,
        batch_size,
    ):
        end_index = start_index + batch_size

        batches.append(
            observation_indices[start_index:end_index]
        )

    return batches


def optimize_with_full_batch_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.1,
    number_of_epochs=1_000,
):
    """Estimate parameters using handcrafted full-batch GD."""

    comparison_array = _validate_optimization_inputs(
        number_of_items,
        comparisons,
        learning_rate,
        number_of_epochs,
    )

    start_time = perf_counter()

    (
        estimated_parameters,
        nll_history,
        parameter_history,
    ) = estimate_parameters_with_gradient_descent_trace(
        number_of_items=number_of_items,
        comparisons=comparison_array,
        learning_rate=learning_rate,
        maximum_iterations=number_of_epochs,
    )

    runtime_seconds = perf_counter() - start_time

    return OptimizationResult(
        optimizer_name="Manual full-batch GD",
        parameters=estimated_parameters,
        nll_history=np.asarray(nll_history),
        parameter_history=np.asarray(parameter_history),
        update_count_history=np.arange(
            number_of_epochs + 1
        ),
        runtime_seconds=runtime_seconds,
    )


def _optimize_with_batches(
    optimizer_name,
    number_of_items,
    comparisons,
    learning_rate,
    number_of_epochs,
    batch_size,
    random_seed,
    shuffle,
    momentum=0.0,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
):
    """Run a handcrafted batch-based optimization algorithm."""

    comparison_array = _validate_optimization_inputs(
        number_of_items,
        comparisons,
        learning_rate,
        number_of_epochs,
    )

    if not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be an integer")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if batch_size > len(comparison_array):
        raise ValueError(
            "batch_size cannot exceed the dataset size"
        )

    random_generator = np.random.default_rng(
        random_seed
    )

    (
        parameters,
        nll_history,
        parameter_history,
        update_count_history,
    ) = _initialise_optimization_history(
        number_of_items,
        comparison_array,
    )

    velocity = np.zeros_like(parameters)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)

    number_of_updates = 0
    start_time = perf_counter()

    for _ in range(number_of_epochs):
        epoch_batches = _create_epoch_batches(
            number_of_observations=len(
                comparison_array
            ),
            batch_size=batch_size,
            random_generator=random_generator,
            shuffle=shuffle,
        )

        for batch_indices in epoch_batches:
            comparison_batch = comparison_array[
                batch_indices
            ]

            gradient = (
                average_negative_log_likelihood_gradient(
                    parameters,
                    comparison_batch,
                )
            )

            number_of_updates += 1

            if optimizer_name in {
                "Manual SGD",
                "Manual mini-batch SGD",
            }:
                parameters = (
                    parameters
                    - learning_rate * gradient
                )

            elif optimizer_name == "Manual momentum SGD":
                velocity = (
                    momentum * velocity
                    + gradient
                )

                parameters = (
                    parameters
                    - learning_rate * velocity
                )

            elif optimizer_name == "Manual Adam":
                first_moment = (
                    beta_1 * first_moment
                    + (1.0 - beta_1) * gradient
                )

                second_moment = (
                    beta_2 * second_moment
                    + (1.0 - beta_2)
                    * np.square(gradient)
                )

                corrected_first_moment = (
                    first_moment
                    / (
                        1.0
                        - beta_1 ** number_of_updates
                    )
                )

                corrected_second_moment = (
                    second_moment
                    / (
                        1.0
                        - beta_2 ** number_of_updates
                    )
                )

                parameters = parameters - (
                    learning_rate
                    * corrected_first_moment
                    / (
                        np.sqrt(
                            corrected_second_moment
                        )
                        + epsilon
                    )
                )

            else:
                raise ValueError(
                    f"unknown optimizer: {optimizer_name}"
                )

            parameters = center_parameters(parameters)

        current_nll = average_negative_log_likelihood(
            parameters,
            comparison_array,
        )

        nll_history.append(current_nll)
        parameter_history.append(parameters.copy())
        update_count_history.append(number_of_updates)

    runtime_seconds = perf_counter() - start_time

    return OptimizationResult(
        optimizer_name=optimizer_name,
        parameters=parameters,
        nll_history=np.asarray(nll_history),
        parameter_history=np.asarray(
            parameter_history
        ),
        update_count_history=np.asarray(
            update_count_history
        ),
        runtime_seconds=runtime_seconds,
    )


def optimize_with_stochastic_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.01,
    number_of_epochs=100,
    random_seed=0,
    shuffle=True,
):
    """Estimate parameters using one observation per update."""

    return _optimize_with_batches(
        optimizer_name="Manual SGD",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        number_of_epochs=number_of_epochs,
        batch_size=1,
        random_seed=random_seed,
        shuffle=shuffle,
    )


def optimize_with_minibatch_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.1,
    number_of_epochs=100,
    batch_size=64,
    random_seed=0,
    shuffle=True,
):
    """Estimate parameters using mini-batch SGD."""

    return _optimize_with_batches(
        optimizer_name="Manual mini-batch SGD",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        shuffle=shuffle,
    )


def optimize_with_momentum_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.05,
    number_of_epochs=100,
    batch_size=64,
    momentum=0.9,
    random_seed=0,
    shuffle=True,
):
    """Estimate parameters using mini-batch momentum SGD."""

    if not np.isfinite(momentum):
        raise ValueError("momentum must be finite")

    if not 0 <= momentum < 1:
        raise ValueError(
            "momentum must be in the interval [0, 1)"
        )

    return _optimize_with_batches(
        optimizer_name="Manual momentum SGD",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        shuffle=shuffle,
        momentum=momentum,
    )


def optimize_with_adam(
    number_of_items,
    comparisons,
    learning_rate=0.01,
    number_of_epochs=100,
    batch_size=64,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    random_seed=0,
    shuffle=True,
):
    """Estimate parameters using handcrafted Adam."""

    if not 0 <= beta_1 < 1:
        raise ValueError(
            "beta_1 must be in the interval [0, 1)"
        )

    if not 0 <= beta_2 < 1:
        raise ValueError(
            "beta_2 must be in the interval [0, 1)"
        )

    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError(
            "epsilon must be a positive finite number"
        )

    return _optimize_with_batches(
        optimizer_name="Manual Adam",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        random_seed=random_seed,
        shuffle=shuffle,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )