"""Reusable synthetic-data utilities for CDM paper reproduction."""

from itertools import combinations

import torch

from context_dependent_model import center_cdm_parameters


def generate_paper_cdm_parameters(
    number_of_items: int,
    *,
    random_seed: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate standard-normal factors and centred CDM interactions."""

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least 2.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    shape = (number_of_items, number_of_items)

    target_vectors = torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
    ).to(device)

    context_vectors = torch.randn(
        shape,
        generator=generator,
        dtype=dtype,
    ).to(device)

    raw_interactions = target_vectors @ context_vectors.T

    true_interactions = center_cdm_parameters(
        raw_interactions
    )

    return (
        target_vectors,
        context_vectors,
        true_interactions,
    )


def sample_uniform_subset_choice_sets(
    number_of_items: int,
    number_of_observations: int,
    *,
    random_seed: int,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample subsets uniformly and return padded sets with valid-item masks."""

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least 2.")

    if number_of_observations < 1:
        raise ValueError("number_of_observations must be positive.")

    possible_choice_sets = tuple(
        choice_set
        for size in range(2, number_of_items + 1)
        for choice_set in combinations(
            range(number_of_items),
            size,
        )
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    sampled_set_indices = torch.randint(
        high=len(possible_choice_sets),
        size=(number_of_observations,),
        generator=generator,
    )

    choice_sets = torch.full(
        (number_of_observations, number_of_items),
        fill_value=-1,
        dtype=torch.long,
    )

    for row, set_index in enumerate(
        sampled_set_indices.tolist()
    ):
        current_choice_set = possible_choice_sets[set_index]

        choice_sets[row, :len(current_choice_set)] = (
            torch.tensor(
                current_choice_set,
                dtype=torch.long,
            )
        )

    choice_mask = choice_sets.ge(0)

    return (
        choice_sets.to(device),
        choice_mask.to(device),
    )