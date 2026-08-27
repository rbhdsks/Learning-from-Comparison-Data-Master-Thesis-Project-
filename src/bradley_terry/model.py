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


def _as_comparison_arrays(winners,losers,number_of_items):
    """Convert winners and losers into valid integer arrays."""
    winner_array=np.asarray(winners)
    loser_array=np.asarray(losers)

    if winner_array.ndim!=1 or loser_array.ndim!=1:
        raise ValueError(
            "winners and losers must be one-dimensional arrays"
        )

    if winner_array.size==0:
        raise ValueError("at least one comparison is required")

    if winner_array.shape!=loser_array.shape:
        raise ValueError(
            "winners and losers must have the same length"
        )

    if not np.issubdtype(winner_array.dtype,np.integer):
        raise TypeError("winners must contain integer item indices")

    if not np.issubdtype(loser_array.dtype,np.integer):
        raise TypeError("losers must contain integer item indices")

    winner_array=winner_array.astype(int,copy=False)
    loser_array=loser_array.astype(int,copy=False)

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

    return winner_array,loser_array


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
    winners,
    losers
):
    """Calculate the average NLL of the observed comparisons."""
    parameter_array=_as_parameter_array(parameters)

    winner_array,loser_array=_as_comparison_arrays(
        winners,
        losers,
        len(parameter_array)
    )

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
    winners,
    losers
):
    """Calculate the gradient of the average NLL."""
    parameter_array=_as_parameter_array(parameters)

    winner_array,loser_array=_as_comparison_arrays(
        winners,
        losers,
        len(parameter_array)
    )

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