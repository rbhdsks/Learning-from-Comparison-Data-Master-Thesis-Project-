"""Simulation of pairwise-comparison data."""

import numpy as np

from .model import bradley_terry_probability


def generate_uniform_comparison_data(
    true_parameters,
    number_of_comparisons,
    rng
):
    """Generate comparisons by sampling every item uniformly."""
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

    if not isinstance(
        number_of_comparisons,
        (int,np.integer)
    ):
        raise TypeError(
            "number_of_comparisons must be an integer"
        )

    if number_of_comparisons<=0:
        raise ValueError(
            "number_of_comparisons must be positive"
        )

    if not isinstance(rng,np.random.Generator):
        raise TypeError(
            "rng must be created using np.random.default_rng()"
        )

    number_of_items=len(parameter_array)

    comparisons=[]

    for _ in range(number_of_comparisons):
        item_i,item_j=rng.choice(
            number_of_items,
            size=2,
            replace=False
        )

        probability_i_wins=bradley_terry_probability(
            item_i,
            item_j,
            parameter_array
        )

        if rng.random()<probability_i_wins:
            winner,loser=item_i,item_j
        else:
            winner,loser=item_j,item_i

        comparisons.append(
            (int(winner),int(loser))
        )

    return comparisons