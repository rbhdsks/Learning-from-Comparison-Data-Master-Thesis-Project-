"""Evaluation metrics for Context-Dependent Models."""

from __future__ import annotations

import torch

from .model import center_cdm_parameters


def cdm_parameter_recovery_error(
    true_interactions: torch.Tensor,
    estimated_interactions: torch.Tensor,
) -> float:
    """Calculate RMSE over the off-diagonal CDM parameters."""

    if true_interactions.shape != estimated_interactions.shape:
        raise ValueError(
            "True and estimated matrices must have the same shape."
        )

    if (
        true_interactions.ndim != 2
        or true_interactions.shape[0] != true_interactions.shape[1]
    ):
        raise ValueError("Interaction matrices must be square.")

    true_centered = center_cdm_parameters(
        true_interactions
    )

    estimated_centered = center_cdm_parameters(
        estimated_interactions
    )

    number_of_items = true_centered.shape[0]

    off_diagonal_mask = ~torch.eye(
        number_of_items,
        device=true_centered.device,
        dtype=torch.bool,
    )

    squared_errors = (
        estimated_centered[off_diagonal_mask]
        - true_centered[off_diagonal_mask]
    ) ** 2

    return torch.sqrt(
        squared_errors.mean()
    ).item()