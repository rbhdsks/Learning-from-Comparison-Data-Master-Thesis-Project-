# Estimate full CDM parameters with optional interaction-matrix initialization.
# Import the existing CDM components required by the reproduction optimizer.

from __future__ import annotations

from time import perf_counter

import torch

from context_dependent_model.estimation import (
    CDMOptimizationResult,
)
from context_dependent_model.model import (
    FullContextDependentModel,
    average_cdm_negative_log_likelihood,
)
from context_dependent_model.simulation import (
    CDMChoiceData,
)
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
    initial_interactions: torch.Tensor | None = None,
) -> CDMOptimizationResult:
    """Fit a full CDM using full-batch Adam and optional initial interactions."""

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

    if len(training_data.choice_sets) == 0:
        raise ValueError("training_data must contain observations.")

    device = training_data.choice_sets.device

    model = FullContextDependentModel(
        number_of_items=number_of_items,
        device=device,
        dtype=dtype,
    )

    if initial_interactions is not None:
        initial_matrix = torch.as_tensor(
            initial_interactions,
            device=device,
            dtype=dtype,
        ).detach()

        if initial_matrix.shape != (
            number_of_items,
            number_of_items,
        ):
            raise ValueError(
                "initial_interactions must have shape "
                "(number_of_items, number_of_items)."
            )

        if not torch.isfinite(initial_matrix).all().item():
            raise ValueError(
                "initial_interactions must contain only finite values."
            )

        with torch.no_grad():
            model.raw_interactions.copy_(initial_matrix)

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

    if not torch.isfinite(torch.tensor(initial_nll)).item():
        raise FloatingPointError("Initial NLL is not finite.")

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

        gradient_norm = (
            model.raw_interactions.grad.norm().item()
        )

        if not torch.isfinite(
            model.raw_interactions.grad
        ).all().item():
            raise FloatingPointError(
                "Non-finite gradient encountered during optimization."
            )

        optimizer.step()

        with torch.no_grad():
            updated_nll = average_cdm_negative_log_likelihood(
                interaction_matrix=model.interaction_matrix,
                choice_sets=training_data.choice_sets,
                chosen_positions=training_data.chosen_positions,
                choice_mask=training_data.choice_mask,
            ).item()

        if not torch.isfinite(torch.tensor(updated_nll)).item():
            raise FloatingPointError(
                "Non-finite NLL encountered during optimization."
            )

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
        interaction_matrix=(
            model.interaction_matrix.detach().clone()
        ),
        nll_history=nll_history,
        gradient_norm_history=gradient_norm_history,
        epochs_completed=epoch,
        runtime_seconds=runtime_seconds,
        converged=converged,
        stopping_reason=stopping_reason,
    )