# Run repeated full-CDM recovery trials and record zero-choice separation.

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bradley_terry.choix_estimation import (
    estimate_top1_parameters_with_choix,
)
from context_dependent_model.model import center_cdm_parameters
from context_dependent_model.simulation import (
    CDMChoiceData,
    generate_cdm_choice_data_on_sets,
)

from .estimation import optimize_full_cdm_with_adam
from .synthetic import sample_uniform_subset_choice_sets


def count_zero_choice_separation_pairs(
    data: CDMChoiceData,
    number_of_items: int,
) -> int:
    """Count observed target-context pairs with zero target choices."""

    item_indices = torch.arange(
        number_of_items,
        device=data.choice_sets.device,
    )

    presence = (
        (
            data.choice_sets.unsqueeze(-1)
            == item_indices
        )
        & data.choice_mask.bool().unsqueeze(-1)
    ).any(dim=1).to(torch.int64)

    cooccurrence_counts = presence.T @ presence

    target_choice_counts = torch.zeros(
        (number_of_items, number_of_items),
        dtype=torch.int64,
        device=presence.device,
    )

    target_choice_counts.index_add_(
        0,
        data.chosen_items,
        presence,
    )

    off_diagonal = ~torch.eye(
        number_of_items,
        dtype=torch.bool,
        device=presence.device,
    )

    separation_pairs = (
        off_diagonal
        & cooccurrence_counts.gt(0)
        & target_choice_counts.eq(0)
    )

    return int(separation_pairs.sum().item())


def run_paper_cdm_recovery_scaling(
    true_interactions: torch.Tensor,
    observation_counts: tuple[int, ...],
    *,
    number_of_runs: int = 10,
    base_seed: int = 20260831,
    learning_rate: float = 0.05,
    gradient_tolerance: float = 1e-5,
    nll_tolerance: float = 1e-9,
    patience: int = 50,
    maximum_epochs: int = 5_000,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fit nested samples with Luce initialization and recovery diagnostics."""

    counts = tuple(sorted(set(observation_counts)))

    if not counts or any(
        not isinstance(count, (int, np.integer)) or count < 1
        for count in counts
    ):
        raise ValueError("Observation counts must be positive integers.")

    if number_of_runs < 1:
        raise ValueError("number_of_runs must be positive.")

    truth = center_cdm_parameters(
        true_interactions.detach().clone()
    )

    if not torch.isfinite(truth).all().item():
        raise ValueError("True interactions must be finite.")

    number_of_items = truth.shape[0]
    device = truth.device

    off_diagonal = ~torch.eye(
        number_of_items,
        dtype=torch.bool,
        device=device,
    )

    output_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None else None
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_fits = number_of_runs * len(counts)

    for run_index in range(number_of_runs):
        choice_set_seed = base_seed + 10_000 + 2 * run_index
        choice_outcome_seed = choice_set_seed + 1

        choice_sets, choice_mask = sample_uniform_subset_choice_sets(
            number_of_items=number_of_items,
            number_of_observations=max(counts),
            random_seed=choice_set_seed,
            device=device,
        )

        full_data = generate_cdm_choice_data_on_sets(
            interaction_matrix=truth,
            choice_sets=choice_sets,
            choice_mask=choice_mask,
            random_seed=choice_outcome_seed,
        )

        for observation_count in counts:
            data = CDMChoiceData(
                choice_sets=full_data.choice_sets[:observation_count],
                choice_mask=full_data.choice_mask[:observation_count],
                chosen_positions=(
                    full_data.chosen_positions[:observation_count]
                ),
            )

            separation_count = count_zero_choice_separation_pairs(
                data,
                number_of_items,
            )

            row = {
                "Run": run_index + 1,
                "Number of items": number_of_items,
                "Number of observations": observation_count,
                "Choice-set seed": choice_set_seed,
                "Choice-outcome seed": choice_outcome_seed,
                "Generated pool size": max(counts),
                "Zero-choice separation pairs": separation_count,
                "Zero-choice separation detected": separation_count > 0,
                "Fit status": "pending",
                "Error message": "",
                "Squared parameter error": np.nan,
                "Parameter MSE": np.nan,
                "Parameter RMSE": np.nan,
                "Maximum absolute parameter error": np.nan,
                "Initial training NLL": np.nan,
                "Final training NLL": np.nan,
                "Epochs": np.nan,
                "CDM runtime (seconds)": np.nan,
                "Optimizer converged": None,
                "Stopping reason": "",
            }

            print(
                f"Fit {len(rows) + 1}/{total_fits} | "
                f"run={run_index + 1} | "
                f"observations={observation_count:,} | "
                f"separation pairs={separation_count}",
                flush=True,
            )

            observations = [
                (
                    int(chosen_item),
                    tuple(choice_set[mask].detach().cpu().tolist()),
                )
                for chosen_item, choice_set, mask in zip(
                    data.chosen_items.detach().cpu().tolist(),
                    data.choice_sets,
                    data.choice_mask.bool(),
                )
            ]

            try:
                luce_parameters = estimate_top1_parameters_with_choix(
                    number_of_items=number_of_items,
                    observations=observations,
                    alpha=0.0,
                    maximum_iterations=1_000,
                    tolerance=1e-8,
                )

                luce_tensor = torch.as_tensor(
                    luce_parameters,
                    dtype=truth.dtype,
                    device=device,
                )

                initial_interactions = center_cdm_parameters(
                    -luce_tensor.unsqueeze(0).expand(
                        number_of_items,
                        number_of_items,
                    )
                )

                fit = optimize_full_cdm_with_adam(
                    number_of_items=number_of_items,
                    training_data=data,
                    initial_interactions=initial_interactions,
                    learning_rate=learning_rate,
                    gradient_tolerance=gradient_tolerance,
                    nll_tolerance=nll_tolerance,
                    patience=patience,
                    maximum_epochs=maximum_epochs,
                    dtype=truth.dtype,
                )

                estimate = center_cdm_parameters(
                    fit.interaction_matrix.detach()
                )

                errors = (estimate - truth)[off_diagonal]

                if not torch.isfinite(errors).all().item():
                    raise ValueError("The fitted matrix is not finite.")

                squared_error = errors.square()

                row.update({
                    "Fit status": "completed",
                    "Squared parameter error": squared_error.sum().item(),
                    "Parameter MSE": squared_error.mean().item(),
                    "Parameter RMSE": (
                        squared_error.mean().sqrt().item()
                    ),
                    "Maximum absolute parameter error": (
                        errors.abs().max().item()
                    ),
                    "Initial training NLL": fit.nll_history[0],
                    "Final training NLL": fit.nll_history[-1],
                    "Epochs": fit.epochs_completed,
                    "CDM runtime (seconds)": fit.runtime_seconds,
                    "Optimizer converged": fit.converged,
                    "Stopping reason": fit.stopping_reason,
                })

            except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                row["Fit status"] = "failed"
                row["Error message"] = (
                    f"{type(error).__name__}: {error}"
                )
                print(f"  Recorded failure: {error}", flush=True)

            rows.append(row)

            if output_path is not None:
                pd.DataFrame(rows).to_csv(output_path, index=False)

    return pd.DataFrame(rows)