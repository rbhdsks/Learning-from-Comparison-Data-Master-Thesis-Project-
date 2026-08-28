"""Core mathematics for the Bradley-Terry model."""

import numpy as np


def _as_parameter_array(parameters):
    """Convert parameters into a valid one-dimensional array."""
    parameter_array=np.asarray(parameters,dtype=float)

    if parameter_array.ndim!=1:
        raise ValueError("parameters must be a one-dimensional array")

    if parameter_array.size<2:
        raise ValueError("at least two parameters are required")

    if not np.all(np.isfinite(parameter_array)):
        raise ValueError("parameters must contain only finite values")

    return parameter_array


def _as_comparison_array(comparisons,number_of_items):
    """Convert (winner, loser) pairs into a valid integer array."""
    comparison_array=np.asarray(comparisons)

    if comparison_array.size==0:
        raise ValueError("at least one comparison is required")

    if (
        comparison_array.ndim!=2
        or comparison_array.shape[1]!=2
    ):
        raise ValueError(
            "comparisons must contain (winner, loser) pairs"
        )

    if not np.issubdtype(comparison_array.dtype,np.integer):
        raise TypeError(
            "comparisons must contain integer item indices"
        )

    comparison_array=comparison_array.astype(int,copy=False)
    winner_array=comparison_array[:,0]
    loser_array=comparison_array[:,1]

    if np.any(winner_array<0) or np.any(
        winner_array>=number_of_items
    ):
        raise ValueError(
            "winners contain an item index outside the valid range"
        )

    if np.any(loser_array<0) or np.any(
        loser_array>=number_of_items
    ):
        raise ValueError(
            "losers contain an item index outside the valid range"
        )

    if np.any(winner_array==loser_array):
        raise ValueError("an item cannot be compared with itself")

    return comparison_array


def sigmoid(value):
    """Convert a parameter difference into a probability."""
    clipped_value=np.clip(value,-500.0,500.0)

    return 1.0/(1.0+np.exp(-clipped_value))


def center_parameters(parameters):
    """Shift parameters so that their mean becomes zero."""
    parameter_array=_as_parameter_array(parameters)

    return parameter_array-np.mean(parameter_array)


def bradley_terry_probability(item_i,item_j,parameters):
    """Calculate the probability that item_i defeats item_j."""
    parameter_array=_as_parameter_array(parameters)
    number_of_items=len(parameter_array)

    if not isinstance(item_i,(int,np.integer)):
        raise TypeError("item_i must be an integer index")

    if not isinstance(item_j,(int,np.integer)):
        raise TypeError("item_j must be an integer index")

    if not 0<=item_i<number_of_items:
        raise ValueError("item_i is outside the valid range")

    if not 0<=item_j<number_of_items:
        raise ValueError("item_j is outside the valid range")

    if item_i==item_j:
        raise ValueError("an item cannot be compared with itself")

    parameter_difference=(
        parameter_array[item_i]-parameter_array[item_j]
    )

    return float(sigmoid(parameter_difference))


def average_negative_log_likelihood(
    parameters,
    comparisons
):
    """Calculate the average NLL of the observed comparisons."""
    parameter_array=_as_parameter_array(parameters)

    comparison_array=_as_comparison_array(
        comparisons,
        len(parameter_array)
    )

    winner_array=comparison_array[:,0]
    loser_array=comparison_array[:,1]

    parameter_differences=(
        parameter_array[winner_array]
        -parameter_array[loser_array]
    )

    negative_log_probabilities=np.logaddexp(
        0.0,
        -parameter_differences
    )

    return float(np.mean(negative_log_probabilities))


def average_negative_log_likelihood_gradient(
    parameters,
    comparisons
):
    """Calculate the gradient of the average NLL."""
    parameter_array=_as_parameter_array(parameters)

    comparison_array=_as_comparison_array(
        comparisons,
        len(parameter_array)
    )

    winner_array=comparison_array[:,0]
    loser_array=comparison_array[:,1]

    parameter_differences=(
        parameter_array[winner_array]
        -parameter_array[loser_array]
    )

    winner_probabilities=sigmoid(parameter_differences)
    winner_contributions=winner_probabilities-1.0

    gradient=np.zeros_like(parameter_array)

    np.add.at(
        gradient,
        winner_array,
        winner_contributions
    )

    np.add.at(
        gradient,
        loser_array,
        -winner_contributions
    )

    return gradient/len(winner_array)