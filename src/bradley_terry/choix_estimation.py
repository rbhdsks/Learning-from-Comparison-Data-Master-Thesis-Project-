"""Parameter estimation using the Choix library."""

from collections.abc import Sequence

import choix
import numpy as np

from .model import center_parameters


PairwiseComparison = tuple[int, int]
ChoiceObservation = tuple[int, tuple[int, ...]]


def estimate_pairwise_parameters_with_choix(
    number_of_items: int,
    comparisons: Sequence[PairwiseComparison],
    alpha: float = 0.0,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Estimate BTL parameters from pairwise comparisons."""

    estimated_parameters = choix.ilsr_pairwise(
        n_items=number_of_items,
        data=comparisons,
        alpha=alpha,
        max_iter=maximum_iterations,
        tol=tolerance,
    )

    return center_parameters(estimated_parameters)


def convert_choice_observations_to_choix_top1(
    observations: Sequence[ChoiceObservation],
) -> list[tuple[int, tuple[int, ...]]]:
    """Convert (chosen item, choice set) observations to Choix format."""

    top1_observations = []

    for chosen_item, choice_set in observations:
        if chosen_item not in choice_set:
            raise ValueError(
                "chosen item must be present in the choice set"
            )

        losing_items = tuple(
            item
            for item in choice_set
            if item != chosen_item
        )

        top1_observations.append(
            (chosen_item, losing_items)
        )

    return top1_observations


def estimate_top1_parameters_with_choix(
    number_of_items: int,
    observations: Sequence[ChoiceObservation],
    alpha: float = 0.0,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Estimate MNL/Luce parameters from general choice observations."""

    top1_observations = (
        convert_choice_observations_to_choix_top1(
            observations
        )
    )

    estimated_parameters = choix.ilsr_top1(
        n_items=number_of_items,
        data=top1_observations,
        alpha=alpha,
        max_iter=maximum_iterations,
        tol=tolerance,
    )

    return center_parameters(estimated_parameters)