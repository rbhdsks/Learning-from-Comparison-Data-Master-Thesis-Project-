"""Synthetic data generation for Context-Dependent Models."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch

from .model import (
    cdm_choice_probabilities,
    center_cdm_parameters,
)


@dataclass
class CDMChoiceData:
    """Stores padded choice sets and their observed choices."""

    choice_sets: torch.Tensor
    choice_mask: torch.Tensor
    chosen_positions: torch.Tensor

    @property
    def chosen_items(self) -> torch.Tensor:
        """Return the item selected in every observation."""

        return self.choice_sets.gather(
            dim=1,
            index=self.chosen_positions.unsqueeze(1),
        ).squeeze(1)


def generate_cdm_parameters(
    number_of_items: int,
    *,
    parameter_scale: float = 0.4,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate a centered full-CDM interaction matrix."""

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least two.")

    if parameter_scale <= 0:
        raise ValueError("parameter_scale must be positive.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    raw_parameters = parameter_scale * torch.randn(
        number_of_items,
        number_of_items,
        generator=generator,
        dtype=dtype,
    )

    raw_parameters = raw_parameters.to(device)

    return center_cdm_parameters(raw_parameters)


def generate_uniform_choice_sets(
    number_of_items: int,
    number_of_observations: int,
    choice_set_sizes: Sequence[int],
    *,
    random_seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate uniformly sampled choice sets of different sizes."""

    if number_of_observations < 1:
        raise ValueError("number_of_observations must be positive.")

    sizes = tuple(int(size) for size in choice_set_sizes)

    if not sizes:
        raise ValueError("choice_set_sizes cannot be empty.")

    if min(sizes) < 2 or max(sizes) > number_of_items:
        raise ValueError(
            "Every choice-set size must be between 2 and number_of_items."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    sampled_size_indices = torch.randint(
        low=0,
        high=len(sizes),
        size=(number_of_observations,),
        generator=generator,
    )

    maximum_size = max(sizes)

    choice_sets = torch.full(
        (number_of_observations, maximum_size),
        fill_value=-1,
        dtype=torch.long,
    )

    choice_mask = torch.zeros(
        (number_of_observations, maximum_size),
        dtype=torch.bool,
    )

    for observation_index, size_index in enumerate(
        sampled_size_indices.tolist()
    ):
        current_size = sizes[size_index]

        selected_items = torch.randperm(
            number_of_items,
            generator=generator,
        )[:current_size]

        choice_sets[
            observation_index,
            :current_size,
        ] = selected_items

        choice_mask[
            observation_index,
            :current_size,
        ] = True

    return (
        choice_sets.to(device),
        choice_mask.to(device),
    )


def generate_cdm_choice_data(
    interaction_matrix: torch.Tensor,
    number_of_observations: int,
    choice_set_sizes: Sequence[int],
    *,
    random_seed: int = 0,
) -> CDMChoiceData:
    """Generate choices from a full CDM interaction matrix."""

    number_of_items = interaction_matrix.shape[0]

    choice_sets, choice_mask = generate_uniform_choice_sets(
        number_of_items=number_of_items,
        number_of_observations=number_of_observations,
        choice_set_sizes=choice_set_sizes,
        random_seed=random_seed,
        device=interaction_matrix.device,
    )

    with torch.no_grad():
        probabilities = cdm_choice_probabilities(
            interaction_matrix=interaction_matrix,
            choice_sets=choice_sets,
            choice_mask=choice_mask,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed + 1)

    chosen_positions = torch.multinomial(
        probabilities.cpu(),
        num_samples=1,
        generator=generator,
    ).squeeze(1)

    return CDMChoiceData(
        choice_sets=choice_sets,
        choice_mask=choice_mask,
        chosen_positions=chosen_positions.to(
            interaction_matrix.device
        ),
    )
def split_cdm_choice_data(
    data: CDMChoiceData,
    *,
    train_fraction: float = 0.8,
    random_seed: int = 0,
) -> tuple[CDMChoiceData, CDMChoiceData]:
    """Randomly split CDM observations into training and test data."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")

    number_of_observations = data.choice_sets.shape[0]

    if number_of_observations < 2:
        raise ValueError("At least two observations are required.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    shuffled_indices = torch.randperm(
        number_of_observations,
        generator=generator,
    )

    number_of_training_observations = int(
        train_fraction * number_of_observations
    )

    training_indices = shuffled_indices[
        :number_of_training_observations
    ].to(data.choice_sets.device)

    test_indices = shuffled_indices[
        number_of_training_observations:
    ].to(data.choice_sets.device)

    training_data = CDMChoiceData(
        choice_sets=data.choice_sets[training_indices],
        choice_mask=data.choice_mask[training_indices],
        chosen_positions=data.chosen_positions[training_indices],
    )

    test_data = CDMChoiceData(
        choice_sets=data.choice_sets[test_indices],
        choice_mask=data.choice_mask[test_indices],
        chosen_positions=data.chosen_positions[test_indices],
    )

    return training_data, test_data