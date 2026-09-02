"""Probability, likelihood and gradient functions for MNL and CDM."""
from collections.abc import Sequence
import numpy as np
ChoiceObservation = tuple[int, tuple[int, ...]]


def _validate_choice_set(
    choice_set: Sequence[int],
    number_of_items: int,
) -> np.ndarray:
    """Validate a choice set and return it as an integer array."""

    items = np.asarray(choice_set, dtype=int)

    if items.ndim != 1:
        raise ValueError("choice_set must be one-dimensional")

    if len(items) < 2:
        raise ValueError("choice_set must contain at least two items")

    if len(np.unique(items)) != len(items):
        raise ValueError("choice_set cannot contain repeated items")

    if np.any(items < 0) or np.any(items >= number_of_items):
        raise ValueError("choice_set contains an invalid item index")

    return items


def _softmax(values: np.ndarray) -> np.ndarray:
    """Calculate numerically stable softmax probabilities."""

    shifted_values = values - np.max(values)
    exponentials = np.exp(shifted_values)

    return exponentials / exponentials.sum()


def center_mnl_parameters(
    parameters: np.ndarray,
) -> np.ndarray:
    """Shift MNL parameters so their mean is zero."""

    parameters = np.asarray(parameters, dtype=float)

    if parameters.ndim != 1:
        raise ValueError("MNL parameters must be one-dimensional")

    return parameters - parameters.mean()


def center_cdm_parameters(
    parameters: np.ndarray,
) -> np.ndarray:
    """Set the diagonal to zero and centre all off-diagonal parameters."""

    parameters = np.asarray(
        parameters,
        dtype=float,
    ).copy()

    if (
        parameters.ndim != 2
        or parameters.shape[0] != parameters.shape[1]
    ):
        raise ValueError(
            "CDM parameters must be a square matrix"
        )

    np.fill_diagonal(parameters, 0.0)

    off_diagonal_mask = ~np.eye(
        parameters.shape[0],
        dtype=bool,
    )

    parameters[off_diagonal_mask] -= (
        parameters[off_diagonal_mask].mean()
    )

    return parameters


def mnl_choice_probabilities(
    parameters: np.ndarray,
    choice_set: Sequence[int],
) -> np.ndarray:
    """Calculate context-free choice probabilities."""

    parameters = np.asarray(parameters, dtype=float)

    if parameters.ndim != 1:
        raise ValueError("MNL parameters must be one-dimensional")

    items = _validate_choice_set(
        choice_set,
        len(parameters),
    )

    return _softmax(parameters[items])


def cdm_choice_utilities(
    parameters: np.ndarray,
    choice_set: Sequence[int],
) -> np.ndarray:
    """Calculate CDM utilities for the items in one choice set."""

    parameters = np.asarray(parameters, dtype=float)

    if (
        parameters.ndim != 2
        or parameters.shape[0] != parameters.shape[1]
    ):
        raise ValueError(
            "CDM parameters must be a square matrix"
        )

    items = _validate_choice_set(
        choice_set,
        parameters.shape[0],
    )

    contextual_parameters = parameters[
        np.ix_(items, items)
    ].copy()

    # An item cannot provide context to itself.
    np.fill_diagonal(contextual_parameters, 0.0)

    return contextual_parameters.sum(axis=1)


def cdm_choice_probabilities(
    parameters: np.ndarray,
    choice_set: Sequence[int],
) -> np.ndarray:
    """Calculate CDM choice probabilities."""

    utilities = cdm_choice_utilities(
        parameters,
        choice_set,
    )

    return _softmax(utilities)


def average_mnl_negative_log_likelihood(
    parameters: np.ndarray,
    observations: Sequence[ChoiceObservation],
) -> float:
    """Calculate average MNL negative log-likelihood."""

    if len(observations) == 0:
        raise ValueError("observations cannot be empty")

    total_nll = 0.0

    for chosen_item, choice_set in observations:
        items = _validate_choice_set(
            choice_set,
            len(parameters),
        )

        chosen_positions = np.flatnonzero(
            items == chosen_item
        )

        if len(chosen_positions) != 1:
            raise ValueError(
                "chosen item must occur in its choice set"
            )

        probabilities = mnl_choice_probabilities(
            parameters,
            items,
        )

        total_nll -= np.log(
            probabilities[chosen_positions[0]]
        )

    return total_nll / len(observations)


def average_cdm_negative_log_likelihood(
    parameters: np.ndarray,
    observations: Sequence[ChoiceObservation],
) -> float:
    """Calculate average CDM negative log-likelihood."""

    if len(observations) == 0:
        raise ValueError("observations cannot be empty")

    total_nll = 0.0

    for chosen_item, choice_set in observations:
        items = _validate_choice_set(
            choice_set,
            parameters.shape[0],
        )

        chosen_positions = np.flatnonzero(
            items == chosen_item
        )

        if len(chosen_positions) != 1:
            raise ValueError(
                "chosen item must occur in its choice set"
            )

        probabilities = cdm_choice_probabilities(
            parameters,
            items,
        )

        total_nll -= np.log(
            probabilities[chosen_positions[0]]
        )

    return total_nll / len(observations)


def average_mnl_nll_gradient(
    parameters: np.ndarray,
    observations: Sequence[ChoiceObservation],
) -> np.ndarray:
    """Calculate the gradient of average MNL NLL."""

    if len(observations) == 0:
        raise ValueError("observations cannot be empty")

    gradient = np.zeros_like(
        parameters,
        dtype=float,
    )

    for chosen_item, choice_set in observations:
        items = _validate_choice_set(
            choice_set,
            len(parameters),
        )

        probabilities = mnl_choice_probabilities(
            parameters,
            items,
        )

        gradient[items] += probabilities
        gradient[chosen_item] -= 1.0

    return gradient / len(observations)


def average_cdm_nll_gradient(
    parameters: np.ndarray,
    observations: Sequence[ChoiceObservation],
) -> np.ndarray:
    """Calculate the gradient of average CDM NLL."""

    if len(observations) == 0:
        raise ValueError("observations cannot be empty")

    gradient = np.zeros_like(
        parameters,
        dtype=float,
    )

    for chosen_item, choice_set in observations:
        items = _validate_choice_set(
            choice_set,
            parameters.shape[0],
        )

        probabilities = cdm_choice_probabilities(
            parameters,
            items,
        )

        utility_derivatives = probabilities.copy()

        chosen_position = np.flatnonzero(
            items == chosen_item
        )[0]

        utility_derivatives[chosen_position] -= 1.0

        for target_position, target_item in enumerate(items):
            for context_item in items:
                if context_item != target_item:
                    gradient[
                        target_item,
                        context_item,
                    ] += utility_derivatives[
                        target_position
                    ]

    gradient /= len(observations)
    np.fill_diagonal(gradient, 0.0)

    return gradient


def convert_mnl_parameters_to_cdm(
    mnl_parameters: np.ndarray,
) -> np.ndarray:
    """Represent an MNL model using equivalent CDM parameters."""

    mnl_parameters = center_mnl_parameters(
        mnl_parameters
    )

    number_of_items = len(mnl_parameters)

    cdm_parameters = np.zeros(
        (number_of_items, number_of_items),
        dtype=float,
    )

    for target_item in range(number_of_items):
        for context_item in range(number_of_items):
            if target_item != context_item:
                cdm_parameters[
                    target_item,
                    context_item,
                ] = -mnl_parameters[context_item]

    return center_cdm_parameters(cdm_parameters)