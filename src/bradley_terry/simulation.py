"""Simulation of pairwise-comparison data."""

import numpy as np

from .model import sigmoid
from .sampling import (
    build_complete_comparison_edges,
)


def _validate_simulation_inputs(
    true_parameters,
    number_of_comparisons,
    rng,
):
    """Validate inputs shared by comparison simulations."""
    parameter_array = np.asarray(
        true_parameters,
        dtype=float,
    )

    if parameter_array.ndim != 1:
        raise ValueError(
            "true_parameters must be one-dimensional"
        )

    if parameter_array.size < 2:
        raise ValueError(
            "at least two true parameters are required"
        )

    if not np.all(np.isfinite(parameter_array)):
        raise ValueError(
            "true_parameters must contain finite values"
        )

    if not isinstance(
        number_of_comparisons,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_comparisons must be an integer"
        )

    if number_of_comparisons <= 0:
        raise ValueError(
            "number_of_comparisons must be positive"
        )

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be created using np.random.default_rng()"
        )

    return parameter_array


def _validate_comparison_edges(
    edges,
    number_of_items,
):
    """Convert graph edges into a valid integer array."""
    edge_array = np.asarray(edges)

    if edge_array.size == 0:
        raise ValueError(
            "at least one comparison edge is required"
        )

    if (
        edge_array.ndim != 2
        or edge_array.shape[1] != 2
    ):
        raise ValueError(
            "edges must contain pairs of item indices"
        )

    if not np.issubdtype(
        edge_array.dtype,
        np.integer,
    ):
        raise TypeError(
            "edges must contain integer item indices"
        )

    edge_array = edge_array.astype(
        int,
        copy=False,
    )

    if (
        np.any(edge_array < 0)
        or np.any(edge_array >= number_of_items)
    ):
        raise ValueError(
            "edges contain an invalid item index"
        )

    if np.any(
        edge_array[:, 0] == edge_array[:, 1]
    ):
        raise ValueError(
            "an item cannot be compared with itself"
        )

    return edge_array


def generate_comparison_data_from_edges(
    true_parameters,
    edges,
    number_of_comparisons,
    rng,
):
    """Generate (winner, loser) observations from graph edges."""
    parameter_array = _validate_simulation_inputs(
        true_parameters,
        number_of_comparisons,
        rng,
    )

    edge_array = _validate_comparison_edges(
        edges,
        number_of_items=len(parameter_array),
    )

    sampled_edge_indices = rng.integers(
        low=0,
        high=len(edge_array),
        size=number_of_comparisons,
    )

    sampled_edges = edge_array[
        sampled_edge_indices
    ]

    first_items = sampled_edges[:, 0]
    second_items = sampled_edges[:, 1]

    parameter_differences = (
        parameter_array[first_items]
        - parameter_array[second_items]
    )

    first_item_win_probabilities = sigmoid(
        parameter_differences
    )

    first_item_wins = (
        rng.random(number_of_comparisons)
        < first_item_win_probabilities
    )

    winners = np.where(
        first_item_wins,
        first_items,
        second_items,
    )

    losers = np.where(
        first_item_wins,
        second_items,
        first_items,
    )

    return list(zip(
        winners.astype(int).tolist(),
        losers.astype(int).tolist(),
    ))


def generate_uniform_comparison_data(
    true_parameters,
    number_of_comparisons,
    rng,
):
    """Generate data by sampling every possible pair uniformly."""
    parameter_array = np.asarray(
        true_parameters,
        dtype=float,
    )

    if parameter_array.ndim != 1:
        raise ValueError(
            "true_parameters must be one-dimensional"
        )

    complete_edges = (
        build_complete_comparison_edges(
            len(parameter_array)
        )
    )

    return generate_comparison_data_from_edges(
        true_parameters=parameter_array,
        edges=complete_edges,
        number_of_comparisons=number_of_comparisons,
        rng=rng,
    )
def generate_top1_choice_data_on_sets(true_parameters, choice_sets, rng):
    """Sample one BTL winner from each supplied choice set."""
    parameters = np.asarray(true_parameters, dtype=float)
    observations = []

    for choice_set in choice_sets:
        items = tuple(int(item) for item in choice_set)
        utilities = parameters[list(items)]
        weights = np.exp(utilities - utilities.max())
        winner = int(rng.choice(items, p=weights / weights.sum()))
        observations.append((winner, items))

    return observations