"""Reusable simulation helpers for synthetic IIA experiments."""

from itertools import combinations

import torch

from context_dependent_model.simulation import CDMChoiceData


def enumerate_choice_sets(
    number_of_items: int,
    *,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate every choice set containing at least two items."""

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least two.")

    subsets = [
        subset
        for size in range(2, number_of_items + 1)
        for subset in combinations(range(number_of_items), size)
    ]

    choice_sets = torch.full(
        (len(subsets), number_of_items),
        -1,
        dtype=torch.long,
        device=device,
    )

    for row, subset in enumerate(subsets):
        choice_sets[row, :len(subset)] = torch.tensor(
            subset,
            dtype=torch.long,
            device=device,
        )

    return choice_sets, choice_sets.ge(0)


def generate_general_choice_probabilities(
    choice_mask: torch.Tensor,
    *,
    random_seed: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate normalized independent uniform weights for each menu."""

    if choice_mask.ndim != 2 or choice_mask.dtype != torch.bool:
        raise ValueError("choice_mask must be a two-dimensional bool tensor.")

    if not choice_mask.any(dim=1).all().item():
        raise ValueError("Every menu must contain at least one valid item.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    weights = torch.rand(
        choice_mask.shape,
        generator=generator,
        dtype=dtype,
    ).clamp_min(torch.finfo(dtype).tiny)

    weights = weights.to(choice_mask.device)
    weights = weights.masked_fill(~choice_mask, 0.0)

    return weights / weights.sum(dim=1, keepdim=True)


def sample_choices_from_probability_table(
    catalogue_choice_sets: torch.Tensor,
    catalogue_choice_mask: torch.Tensor,
    probability_table: torch.Tensor,
    menu_indices: torch.Tensor,
    *,
    random_seed: int,
) -> CDMChoiceData:
    """Sample one chosen position for each indexed menu."""

    if (
        catalogue_choice_sets.shape != catalogue_choice_mask.shape
        or probability_table.shape != catalogue_choice_sets.shape
    ):
        raise ValueError("Catalogue, mask and probabilities must match.")

    if not torch.isfinite(probability_table).all().item():
        raise ValueError("Probabilities must be finite.")

    if (probability_table < 0).any().item():
        raise ValueError("Probabilities cannot be negative.")

    if (probability_table[~catalogue_choice_mask] != 0).any().item():
        raise ValueError("Padding positions must have zero probability.")

    if not torch.allclose(
        probability_table.sum(dim=1),
        torch.ones(
            probability_table.shape[0],
            dtype=probability_table.dtype,
            device=probability_table.device,
        ),
    ):
        raise ValueError("Each probability row must sum to one.")

    indices = menu_indices.to(
        device=catalogue_choice_sets.device,
        dtype=torch.long,
    )

    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("menu_indices must be a nonempty vector.")

    if (
        (indices < 0).any().item()
        or (indices >= len(catalogue_choice_sets)).any().item()
    ):
        raise ValueError("menu_indices contains an invalid catalogue index.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)

    chosen_positions = torch.multinomial(
        probability_table[indices].detach().cpu(),
        num_samples=1,
        replacement=True,
        generator=generator,
    ).squeeze(1)

    return CDMChoiceData(
        choice_sets=catalogue_choice_sets[indices],
        choice_mask=catalogue_choice_mask[indices],
        chosen_positions=chosen_positions.to(
            catalogue_choice_sets.device
        ),
    )