"""PyTorch estimation functions for full and low-rank CDMs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from .model import (
    FullContextDependentModel,
    LowRankContextDependentModel,
    average_cdm_negative_log_likelihood,
)
from .simulation import CDMChoiceData


@dataclass
class CDMOptimizationResult:
    """Store the output and diagnostics of full CDM optimization."""

    interaction_matrix: torch.Tensor
    nll_history: list[float]
    gradient_norm_history: list[float]
    epochs_completed: int
    runtime_seconds: float
    converged: bool
    stopping_reason: str


@dataclass
class LowRankCDMOptimizationResult:
    """Store the fitted low-rank CDM and its optimization diagnostics."""

    rank: int
    random_seed: int
    target_vectors: torch.Tensor
    context_vectors: torch.Tensor
    interaction_matrix: torch.Tensor
    nll_history: list[float]
    gradient_norm_history: list[float]
    epochs_completed: int
    runtime_seconds: float
    converged: bool
    stopping_reason: str


def optimize_full_cdm_with_adam(
    number_of_items: int,
    training_data: CDMChoiceData,
    *,
    learning_rate: float = 0.05,
    gradient_tolerance: float = 1e-5,
    nll_tolerance: float = 1e-9,
    patience: int = 50,
    maximum_epochs: int = 5_000,
    dtype: torch.dtype = torch.float64,
) -> CDMOptimizationResult:
    """Estimate a full CDM interaction matrix using full-batch Adam."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be positive.")

    if nll_tolerance <= 0:
        raise ValueError("nll_tolerance must be positive.")

    if patience < 1:
        raise ValueError("patience must be positive.")

    device = training_data.choice_sets.device

    model = FullContextDependentModel(
        number_of_items=number_of_items,
        device=device,
        dtype=dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    with torch.no_grad():
        initial_nll = average_cdm_negative_log_likelihood(
            interaction_matrix=model.interaction_matrix,
            choice_sets=training_data.choice_sets,
            chosen_positions=training_data.chosen_positions,
            choice_mask=training_data.choice_mask,
        ).item()

    nll_history = [initial_nll]
    gradient_norm_history = []

    previous_nll = initial_nll
    stable_epochs = 0
    converged = False
    stopping_reason = "maximum epochs reached"

    start_time = perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        optimizer.zero_grad()

        training_nll = average_cdm_negative_log_likelihood(
            interaction_matrix=model.interaction_matrix,
            choice_sets=training_data.choice_sets,
            chosen_positions=training_data.chosen_positions,
            choice_mask=training_data.choice_mask,
        )

        training_nll.backward()

        gradient_norm = model.raw_interactions.grad.norm().item()

        optimizer.step()

        with torch.no_grad():
            updated_nll = average_cdm_negative_log_likelihood(
                interaction_matrix=model.interaction_matrix,
                choice_sets=training_data.choice_sets,
                chosen_positions=training_data.chosen_positions,
                choice_mask=training_data.choice_mask,
            ).item()

        nll_history.append(updated_nll)
        gradient_norm_history.append(gradient_norm)

        if gradient_norm <= gradient_tolerance:
            converged = True
            stopping_reason = "gradient tolerance reached"
            break

        nll_change = abs(previous_nll - updated_nll)

        if nll_change <= nll_tolerance:
            stable_epochs += 1
        else:
            stable_epochs = 0

        if stable_epochs >= patience:
            converged = True
            stopping_reason = "NLL stopped changing"
            break

        previous_nll = updated_nll

    runtime_seconds = perf_counter() - start_time

    return CDMOptimizationResult(
        interaction_matrix=model.interaction_matrix.detach().clone(),
        nll_history=nll_history,
        gradient_norm_history=gradient_norm_history,
        epochs_completed=epoch,
        runtime_seconds=runtime_seconds,
        converged=converged,
        stopping_reason=stopping_reason,
    )


def optimize_low_rank_cdm_with_adam(
    number_of_items: int,
    rank: int,
    training_data: CDMChoiceData,
    *,
    random_seed: int = 0,
    learning_rate: float = 0.05,
    gradient_tolerance: float = 1e-5,
    nll_tolerance: float = 1e-9,
    patience: int = 50,
    maximum_epochs: int = 5_000,
    dtype: torch.dtype = torch.float64,
) -> LowRankCDMOptimizationResult:
    """Estimate a rank-r CDM using full-batch Adam."""

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be positive.")

    if nll_tolerance <= 0:
        raise ValueError("nll_tolerance must be positive.")

    if patience < 1:
        raise ValueError("patience must be positive.")

    if maximum_epochs < 1:
        raise ValueError("maximum_epochs must be positive.")

    model = LowRankContextDependentModel(
        number_of_items=number_of_items,
        rank=rank,
        random_seed=random_seed,
        device=training_data.choice_sets.device,
        dtype=dtype,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    with torch.no_grad():
        initial_nll = average_cdm_negative_log_likelihood(
            interaction_matrix=model.interaction_matrix,
            choice_sets=training_data.choice_sets,
            chosen_positions=training_data.chosen_positions,
            choice_mask=training_data.choice_mask,
        ).item()

    nll_history = [initial_nll]
    gradient_norm_history = []

    previous_nll = initial_nll
    stable_epochs = 0
    converged = False
    stopping_reason = "maximum epochs reached"

    start_time = perf_counter()

    for epoch in range(1, maximum_epochs + 1):
        optimizer.zero_grad()

        training_nll = average_cdm_negative_log_likelihood(
            interaction_matrix=model.interaction_matrix,
            choice_sets=training_data.choice_sets,
            chosen_positions=training_data.chosen_positions,
            choice_mask=training_data.choice_mask,
        )

        training_nll.backward()

        gradient_norm = torch.sqrt(
            sum(
                parameter.grad.square().sum()
                for parameter in model.parameters()
            )
        ).item()

        optimizer.step()

        with torch.no_grad():
            updated_nll = average_cdm_negative_log_likelihood(
                interaction_matrix=model.interaction_matrix,
                choice_sets=training_data.choice_sets,
                chosen_positions=training_data.chosen_positions,
                choice_mask=training_data.choice_mask,
            ).item()

        nll_history.append(updated_nll)
        gradient_norm_history.append(gradient_norm)

        if gradient_norm <= gradient_tolerance:
            converged = True
            stopping_reason = "gradient tolerance reached"
            break

        nll_change = abs(previous_nll - updated_nll)

        if nll_change <= nll_tolerance:
            stable_epochs += 1
        else:
            stable_epochs = 0

        if stable_epochs >= patience:
            converged = True
            stopping_reason = "NLL stopped changing"
            break

        previous_nll = updated_nll

    runtime_seconds = perf_counter() - start_time

    return LowRankCDMOptimizationResult(
        rank=rank,
        random_seed=random_seed,
        target_vectors=model.target_vectors.detach().clone(),
        context_vectors=model.context_vectors.detach().clone(),
        interaction_matrix=model.interaction_matrix.detach().clone(),
        nll_history=nll_history,
        gradient_norm_history=gradient_norm_history,
        epochs_completed=epoch,
        runtime_seconds=runtime_seconds,
        converged=converged,
        stopping_reason=stopping_reason,
    )