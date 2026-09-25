"""Core PyTorch functions for the Context-Dependent Model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def center_cdm_parameters(
    interaction_matrix: torch.Tensor,
) -> torch.Tensor:
    """Set the diagonal to zero and center all off-diagonal entries."""

    if interaction_matrix.ndim != 2:
        raise ValueError("interaction_matrix must be two-dimensional.")

    if interaction_matrix.shape[0] != interaction_matrix.shape[1]:
        raise ValueError("interaction_matrix must be square.")

    number_of_items = interaction_matrix.shape[0]

    off_diagonal_mask = (
        1.0
        - torch.eye(
            number_of_items,
            device=interaction_matrix.device,
            dtype=interaction_matrix.dtype,
        )
    )

    off_diagonal_mean = (
        interaction_matrix * off_diagonal_mask
    ).sum() / off_diagonal_mask.sum()

    return (
        interaction_matrix - off_diagonal_mean
    ) * off_diagonal_mask


def cdm_choice_utilities(
    interaction_matrix: torch.Tensor,
    choice_sets: torch.Tensor,
    choice_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate CDM utilities for a batch of padded choice sets."""

    if choice_sets.ndim != 2:
        raise ValueError("choice_sets must have shape (observations, positions).")

    if choice_sets.dtype != torch.long:
        raise TypeError("choice_sets must use torch.long indices.")

    if choice_mask is None:
        choice_mask = choice_sets >= 0

    if choice_mask.shape != choice_sets.shape:
        raise ValueError("choice_mask and choice_sets must have the same shape.")

    safe_choice_sets = choice_sets.masked_fill(~choice_mask, 0)

    target_indices = safe_choice_sets.unsqueeze(2)
    context_indices = safe_choice_sets.unsqueeze(1)

    pairwise_effects = interaction_matrix[
        target_indices,
        context_indices,
    ]

    maximum_choice_set_size = choice_sets.shape[1]

    different_positions = ~torch.eye(
        maximum_choice_set_size,
        device=choice_sets.device,
        dtype=torch.bool,
    ).unsqueeze(0)

    valid_pairs = (
        choice_mask.unsqueeze(2)
        & choice_mask.unsqueeze(1)
        & different_positions
    )

    utilities = (
        pairwise_effects * valid_pairs.to(pairwise_effects.dtype)
    ).sum(dim=2)

    return utilities.masked_fill(~choice_mask, -torch.inf)


def cdm_choice_probabilities(
    interaction_matrix: torch.Tensor,
    choice_sets: torch.Tensor,
    choice_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate CDM choice probabilities."""

    utilities = cdm_choice_utilities(
        interaction_matrix,
        choice_sets,
        choice_mask,
    )

    return torch.softmax(utilities, dim=1)


def average_cdm_negative_log_likelihood(
    interaction_matrix: torch.Tensor,
    choice_sets: torch.Tensor,
    chosen_positions: torch.Tensor,
    choice_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Calculate the average CDM negative log-likelihood."""

    utilities = cdm_choice_utilities(
        interaction_matrix,
        choice_sets,
        choice_mask,
    )

    return F.cross_entropy(
        utilities,
        chosen_positions,
        reduction="mean",
    )


class FullContextDependentModel(nn.Module):
    """Trainable full-rank Context-Dependent Model."""

    def __init__(
        self,
        number_of_items: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        if number_of_items < 2:
            raise ValueError("number_of_items must be at least two.")

        self.number_of_items = number_of_items

        self.raw_interactions = nn.Parameter(
            torch.zeros(
                number_of_items,
                number_of_items,
                device=device,
                dtype=dtype,
            )
        )

    @property
    def interaction_matrix(self) -> torch.Tensor:
        """Return the identifiable, centered interaction matrix."""

        return center_cdm_parameters(self.raw_interactions)

    def forward(
        self,
        choice_sets: torch.Tensor,
        choice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the utility of every available item."""

        return cdm_choice_utilities(
            self.interaction_matrix,
            choice_sets,
            choice_mask,
        )

    def probabilities(
        self,
        choice_sets: torch.Tensor,
        choice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return choice probabilities for each observation."""

        return torch.softmax(
            self.forward(choice_sets, choice_mask),
            dim=1,
        )

class LowRankContextDependentModel(nn.Module):
    """CDM with learned target and context vectors."""

    def __init__(
        self,
        number_of_items: int,
        rank: int,
        *,
        initial_scale: float = 0.1,
        random_seed: int = 0,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()

        if number_of_items < 2 or rank < 1 or initial_scale <= 0:
            raise ValueError("Check number_of_items, rank, and initial_scale.")

        self.number_of_items = number_of_items
        self.rank = rank

        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        shape = (number_of_items, rank)

        self.target_vectors = nn.Parameter(
            (initial_scale * torch.randn(shape, generator=generator, dtype=dtype))
            .to(device)
        )
        self.context_vectors = nn.Parameter(
            (initial_scale * torch.randn(shape, generator=generator, dtype=dtype))
            .to(device)
        )

    @property
    def interaction_matrix(self) -> torch.Tensor:
        raw_interactions = self.target_vectors @ self.context_vectors.T
        return center_cdm_parameters(raw_interactions)

    def forward(
        self,
        choice_sets: torch.Tensor,
        choice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return cdm_choice_utilities(
            self.interaction_matrix, choice_sets, choice_mask
        )

    def probabilities(
        self,
        choice_sets: torch.Tensor,
        choice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.softmax(self.forward(choice_sets, choice_mask), dim=1)