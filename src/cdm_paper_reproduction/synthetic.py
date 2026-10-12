# Generate synthetic CDM parameters from independent standard-normal factors.

import torch

from context_dependent_model import center_cdm_parameters


def generate_paper_cdm_parameters(
    number_of_items: int,
    *,
    random_seed: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target vectors, context vectors and centred CDM interactions."""

    if number_of_items < 2:
        raise ValueError("number_of_items must be at least two.")

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
    true_interactions = center_cdm_parameters(raw_interactions)

    return target_vectors, context_vectors, true_interactions