"""Parameter estimation using the Choix library."""

from collections.abc import Sequence

import choix
import numpy as np

from .model import center_parameters


PairwiseComparison = tuple[int, int]

# Internal observation format: (chosen item, complete choice set).
ChoiceObservation = tuple[int, tuple[int, ...]]

# Choix top-1 format: (chosen item, losing items).
ChoixTop1Observation = tuple[int, tuple[int, ...]]


def estimate_pairwise_parameters_with_choix(
    number_of_items: int,
    comparisons: Sequence[PairwiseComparison],
    alpha: float = 0.0,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Estimate BTL parameters from pairwise comparisons."""

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least 2")

    if len(comparisons) == 0:
        raise ValueError("comparisons must not be empty")

    estimated_parameters = choix.ilsr_pairwise(
        n_items=number_of_items,
        data=comparisons,
        alpha=alpha,
        max_iter=maximum_iterations,
        tol=tolerance,
    )

    return center_parameters(
        np.asarray(estimated_parameters, dtype=float)
    )


def convert_choice_observations_to_choix_top1(
    observations: Sequence[ChoiceObservation],
) -> list[ChoixTop1Observation]:
    """Convert (chosen item, complete choice set) observations to Choix format."""

    if len(observations) == 0:
        raise ValueError("observations must not be empty")

    top1_observations: list[ChoixTop1Observation] = []

    for chosen_item, choice_set in observations:
        chosen_item = int(chosen_item)
        choice_set = tuple(int(item) for item in choice_set)

        if len(choice_set) < 2:
            raise ValueError(
                "each choice set must contain at least two items"
            )

        if len(set(choice_set)) != len(choice_set):
            raise ValueError(
                "a choice set must not contain duplicate items"
            )

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

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least 2")

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

    return center_parameters(
        np.asarray(estimated_parameters, dtype=float)
    )