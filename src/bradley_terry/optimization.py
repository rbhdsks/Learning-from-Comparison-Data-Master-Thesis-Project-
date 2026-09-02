"""Handcrafted optimization algorithms for the Bradley-Terry model."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .model import (
    average_negative_log_likelihood,
    average_negative_log_likelihood_gradient,
    center_parameters,
)


@dataclass
class OptimizationResult:
    """Store the result and history of one optimization run."""

    optimizer_name: str
    parameters: np.ndarray
    nll_history: np.ndarray
    gradient_norm_history: np.ndarray
    parameter_history: np.ndarray
    update_count_history: np.ndarray
    runtime_seconds: float
    converged: bool
    stopping_reason: str

    @property
    def epochs_completed(self):
        """Return the number of completed data passes."""
        return len(self.nll_history) - 1

    @property
    def updates_completed(self):
        """Return the total number of parameter updates."""
        return int(self.update_count_history[-1])


def _as_comparison_array(
    comparisons,
    number_of_items,
):
    """Convert comparisons into a valid integer array."""

    comparison_array = np.asarray(comparisons)

    if (
        comparison_array.ndim != 2
        or comparison_array.shape[1] != 2
    ):
        raise ValueError(
            "comparisons must contain (winner, loser) pairs"
        )

    if len(comparison_array) == 0:
        raise ValueError(
            "at least one comparison is required"
        )

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


def _validate_common_settings(
    number_of_items,
    learning_rate,
    gradient_tolerance,
    target_nll,
    emergency_maximum_epochs,
):
    """Validate settings shared by all optimizers."""

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

    if gradient_tolerance is not None:
        if (
            not np.isfinite(gradient_tolerance)
            or gradient_tolerance <= 0
        ):
            raise ValueError(
                "gradient_tolerance must be positive"
            )

    if target_nll is not None:
        if (
            not np.isfinite(target_nll)
            or target_nll < 0
        ):
            raise ValueError(
                "target_nll must be non-negative and finite"
            )

    if (
        gradient_tolerance is None
        and target_nll is None
    ):
        raise ValueError(
            "provide a gradient tolerance or target NLL"
        )

    if not isinstance(
        emergency_maximum_epochs,
        (int, np.integer),
    ):
        raise TypeError(
            "emergency_maximum_epochs must be an integer"
        )

    if emergency_maximum_epochs <= 0:
        raise ValueError(
            "emergency_maximum_epochs must be positive"
        )


def _validate_batch_settings(
    batch_size,
    number_of_observations,
    random_seed,
    shuffle,
):
    """Validate settings used by batch-based methods."""

    if not isinstance(
        batch_size,
        (int, np.integer),
    ):
        raise TypeError(
            "batch_size must be an integer"
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if batch_size > number_of_observations:
        raise ValueError(
            "batch_size cannot exceed the dataset size"
        )

    if not isinstance(
        random_seed,
        (int, np.integer),
    ):
        raise TypeError(
            "random_seed must be an integer"
        )

    if not isinstance(
        shuffle,
        (bool, np.bool_),
    ):
        raise TypeError(
            "shuffle must be True or False"
        )


def _initialise_optimization(
    number_of_items,
    comparisons,
):
    """Create zero parameters and initial diagnostics."""

    parameters = np.zeros(
        number_of_items,
        dtype=float,
    )

    initial_nll = average_negative_log_likelihood(
        parameters,
        comparisons,
    )

    initial_gradient = (
        average_negative_log_likelihood_gradient(
            parameters,
            comparisons,
        )
    )

    initial_gradient_norm = np.linalg.norm(
        initial_gradient
    )

    nll_history = [initial_nll]
    gradient_norm_history = [
        initial_gradient_norm
    ]
    parameter_history = [parameters.copy()]
    update_count_history = [0]

    return (
        parameters,
        initial_gradient,
        nll_history,
        gradient_norm_history,
        parameter_history,
        update_count_history,
    )


def _find_stopping_reason(
    current_nll,
    gradient_norm,
    gradient_tolerance,
    target_nll,
):
    """Check whether a requested stopping condition holds."""

    if (
        gradient_tolerance is not None
        and gradient_norm <= gradient_tolerance
    ):
        return "gradient tolerance reached"

    if (
        target_nll is not None
        and current_nll <= target_nll
    ):
        return "target NLL reached"

    return None


def _create_epoch_batches(
    number_of_observations,
    batch_size,
    random_generator,
    shuffle,
):
    """Yield observation-index batches for one epoch."""

    observation_indices = np.arange(
        number_of_observations
    )

    if shuffle:
        observation_indices = (
            random_generator.permutation(
                observation_indices
            )
        )

    for start_index in range(
        0,
        number_of_observations,
        batch_size,
    ):
        end_index = start_index + batch_size

        yield observation_indices[
            start_index:end_index
        ]


def _create_result(
    optimizer_name,
    parameters,
    nll_history,
    gradient_norm_history,
    parameter_history,
    update_count_history,
    runtime_seconds,
    stopping_reason,
):
    """Construct a consistently formatted result."""

    converged = (
        stopping_reason
        != "emergency maximum epochs reached"
    )

    return OptimizationResult(
        optimizer_name=optimizer_name,
        parameters=np.asarray(parameters),
        nll_history=np.asarray(nll_history),
        gradient_norm_history=np.asarray(
            gradient_norm_history
        ),
        parameter_history=np.asarray(
            parameter_history
        ),
        update_count_history=np.asarray(
            update_count_history
        ),
        runtime_seconds=float(runtime_seconds),
        converged=converged,
        stopping_reason=stopping_reason,
    )


def optimize_with_full_batch_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.1,
    gradient_tolerance=1e-6,
    target_nll=None,
    emergency_maximum_epochs=100_000,
):
    """Run full-batch GD until a stopping condition holds."""

    _validate_common_settings(
        number_of_items=number_of_items,
        learning_rate=learning_rate,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )

    comparison_array = _as_comparison_array(
        comparisons,
        number_of_items,
    )

    start_time = perf_counter()

    (
        parameters,
        gradient,
        nll_history,
        gradient_norm_history,
        parameter_history,
        update_count_history,
    ) = _initialise_optimization(
        number_of_items,
        comparison_array,
    )

    current_nll = nll_history[-1]
    gradient_norm = gradient_norm_history[-1]

    stopping_reason = _find_stopping_reason(
        current_nll=current_nll,
        gradient_norm=gradient_norm,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
    )

    number_of_epochs = 0
    number_of_updates = 0

    while (
        stopping_reason is None
        and number_of_epochs
        < emergency_maximum_epochs
    ):
        parameters = (
            parameters
            - learning_rate * gradient
        )

        parameters = center_parameters(
            parameters
        )

        number_of_epochs += 1
        number_of_updates += 1

        current_nll = (
            average_negative_log_likelihood(
                parameters,
                comparison_array,
            )
        )

        gradient = (
            average_negative_log_likelihood_gradient(
                parameters,
                comparison_array,
            )
        )

        gradient_norm = np.linalg.norm(
            gradient
        )

        nll_history.append(current_nll)
        gradient_norm_history.append(
            gradient_norm
        )
        parameter_history.append(
            parameters.copy()
        )
        update_count_history.append(
            number_of_updates
        )

        stopping_reason = _find_stopping_reason(
            current_nll=current_nll,
            gradient_norm=gradient_norm,
            gradient_tolerance=(
                gradient_tolerance
            ),
            target_nll=target_nll,
        )

    if stopping_reason is None:
        stopping_reason = (
            "emergency maximum epochs reached"
        )

    runtime_seconds = (
        perf_counter() - start_time
    )

    return _create_result(
        optimizer_name="Manual full-batch GD",
        parameters=parameters,
        nll_history=nll_history,
        gradient_norm_history=(
            gradient_norm_history
        ),
        parameter_history=parameter_history,
        update_count_history=(
            update_count_history
        ),
        runtime_seconds=runtime_seconds,
        stopping_reason=stopping_reason,
    )


def _run_batch_optimizer(
    algorithm,
    optimizer_name,
    number_of_items,
    comparisons,
    learning_rate,
    batch_size,
    gradient_tolerance,
    target_nll,
    random_seed,
    shuffle,
    emergency_maximum_epochs,
    momentum=0.0,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
):
    """Run one handcrafted stochastic optimizer."""

    _validate_common_settings(
        number_of_items=number_of_items,
        learning_rate=learning_rate,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )

    comparison_array = _as_comparison_array(
        comparisons,
        number_of_items,
    )

    _validate_batch_settings(
        batch_size=batch_size,
        number_of_observations=len(
            comparison_array
        ),
        random_seed=random_seed,
        shuffle=shuffle,
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    start_time = perf_counter()

    (
        parameters,
        full_gradient,
        nll_history,
        gradient_norm_history,
        parameter_history,
        update_count_history,
    ) = _initialise_optimization(
        number_of_items,
        comparison_array,
    )

    velocity = np.zeros_like(parameters)
    first_moment = np.zeros_like(parameters)
    second_moment = np.zeros_like(parameters)

    current_nll = nll_history[-1]
    gradient_norm = gradient_norm_history[-1]

    stopping_reason = _find_stopping_reason(
        current_nll=current_nll,
        gradient_norm=gradient_norm,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
    )

    number_of_epochs = 0
    number_of_updates = 0

    while (
        stopping_reason is None
        and number_of_epochs
        < emergency_maximum_epochs
    ):
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

            batch_gradient = (
                average_negative_log_likelihood_gradient(
                    parameters,
                    comparison_batch,
                )
            )

            number_of_updates += 1

            if algorithm == "sgd":
                parameters = (
                    parameters
                    - learning_rate
                    * batch_gradient
                )

            elif algorithm == "momentum":
                velocity = (
                    momentum * velocity
                    + batch_gradient
                )

                parameters = (
                    parameters
                    - learning_rate * velocity
                )

            elif algorithm == "adam":
                first_moment = (
                    beta_1 * first_moment
                    + (1.0 - beta_1)
                    * batch_gradient
                )

                second_moment = (
                    beta_2 * second_moment
                    + (1.0 - beta_2)
                    * np.square(batch_gradient)
                )

                corrected_first_moment = (
                    first_moment
                    / (
                        1.0
                        - beta_1
                        ** number_of_updates
                    )
                )

                corrected_second_moment = (
                    second_moment
                    / (
                        1.0
                        - beta_2
                        ** number_of_updates
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
                    f"unknown algorithm: {algorithm}"
                )

            parameters = center_parameters(
                parameters
            )

        number_of_epochs += 1

        current_nll = (
            average_negative_log_likelihood(
                parameters,
                comparison_array,
            )
        )

        full_gradient = (
            average_negative_log_likelihood_gradient(
                parameters,
                comparison_array,
            )
        )

        gradient_norm = np.linalg.norm(
            full_gradient
        )

        nll_history.append(current_nll)
        gradient_norm_history.append(
            gradient_norm
        )
        parameter_history.append(
            parameters.copy()
        )
        update_count_history.append(
            number_of_updates
        )

        stopping_reason = _find_stopping_reason(
            current_nll=current_nll,
            gradient_norm=gradient_norm,
            gradient_tolerance=(
                gradient_tolerance
            ),
            target_nll=target_nll,
        )

    if stopping_reason is None:
        stopping_reason = (
            "emergency maximum epochs reached"
        )

    runtime_seconds = (
        perf_counter() - start_time
    )

    return _create_result(
        optimizer_name=optimizer_name,
        parameters=parameters,
        nll_history=nll_history,
        gradient_norm_history=(
            gradient_norm_history
        ),
        parameter_history=parameter_history,
        update_count_history=(
            update_count_history
        ),
        runtime_seconds=runtime_seconds,
        stopping_reason=stopping_reason,
    )


def optimize_with_stochastic_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.01,
    gradient_tolerance=1e-6,
    target_nll=None,
    random_seed=0,
    shuffle=True,
    emergency_maximum_epochs=10_000,
):
    """Run handcrafted SGD using one observation per update."""

    return _run_batch_optimizer(
        algorithm="sgd",
        optimizer_name="Manual SGD",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=1,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        random_seed=random_seed,
        shuffle=shuffle,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )


def optimize_with_minibatch_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.1,
    batch_size=64,
    gradient_tolerance=1e-6,
    target_nll=None,
    random_seed=0,
    shuffle=True,
    emergency_maximum_epochs=10_000,
):
    """Run handcrafted mini-batch SGD."""

    return _run_batch_optimizer(
        algorithm="sgd",
        optimizer_name="Manual mini-batch SGD",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        random_seed=random_seed,
        shuffle=shuffle,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )


def optimize_with_momentum_gradient_descent(
    number_of_items,
    comparisons,
    learning_rate=0.05,
    batch_size=64,
    momentum=0.9,
    gradient_tolerance=1e-6,
    target_nll=None,
    random_seed=0,
    shuffle=True,
    emergency_maximum_epochs=10_000,
):
    """Run handcrafted mini-batch momentum SGD."""

    if (
        not np.isfinite(momentum)
        or not 0 <= momentum < 1
    ):
        raise ValueError(
            "momentum must be in [0, 1)"
        )

    return _run_batch_optimizer(
        algorithm="momentum",
        optimizer_name="Manual momentum SGD",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        random_seed=random_seed,
        shuffle=shuffle,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
        momentum=momentum,
    )


def optimize_with_adam(
    number_of_items,
    comparisons,
    learning_rate=0.01,
    batch_size=64,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    gradient_tolerance=1e-6,
    target_nll=None,
    random_seed=0,
    shuffle=True,
    emergency_maximum_epochs=10_000,
):
    """Run handcrafted mini-batch Adam."""

    if (
        not np.isfinite(beta_1)
        or not 0 <= beta_1 < 1
    ):
        raise ValueError(
            "beta_1 must be in [0, 1)"
        )

    if (
        not np.isfinite(beta_2)
        or not 0 <= beta_2 < 1
    ):
        raise ValueError(
            "beta_2 must be in [0, 1)"
        )

    if (
        not np.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError(
            "epsilon must be positive and finite"
        )

    return _run_batch_optimizer(
        algorithm="adam",
        optimizer_name="Manual Adam",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        random_seed=random_seed,
        shuffle=shuffle,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )