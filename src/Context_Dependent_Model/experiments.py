"""Reusable experiment runners for Context-Dependent Models."""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

import numpy as np
import torch

from bradley_terry import (
    average_top1_negative_log_likelihood,
    estimate_top1_parameters_with_choix,
)

from .estimation import optimize_full_cdm_with_adam
from .metrics import cdm_parameter_recovery_error
from .model import average_cdm_negative_log_likelihood
from .simulation import (
    CDMChoiceData,
    generate_cdm_choice_data,
    generate_cdm_parameters,
    split_cdm_choice_data,
)


def _convert_cdm_data_to_choice_observations(
    data: CDMChoiceData,
) -> list[tuple[int, tuple[int, ...]]]:
    """Convert padded CDM data into explicit choice observations."""

    observations = []

    for choice_set, choice_mask, chosen_item in zip(
        data.choice_sets,
        data.choice_mask,
        data.chosen_items,
    ):
        valid_items = choice_set[
            choice_mask
        ].detach().cpu().tolist()

        observations.append(
            (
                int(chosen_item.detach().cpu().item()),
                tuple(
                    int(item)
                    for item in valid_items
                ),
            )
        )

    return observations


def run_cdm_observation_scaling_experiment(
    true_interactions: torch.Tensor,
    observation_counts: Sequence[int],
    choice_set_sizes: Sequence[int],
    random_seeds: Sequence[int],
    *,
    train_fraction: float = 0.8,
    learning_rate: float = 0.05,
    gradient_tolerance: float = 1e-5,
    nll_tolerance: float = 1e-9,
    patience: int = 50,
    maximum_epochs: int = 5_000,
) -> list[dict]:
    """Run repeated CDM recovery experiments with increasing data."""

    number_of_items = true_interactions.shape[0]
    dtype = true_interactions.dtype

    total_trials = (
        len(observation_counts)
        * len(random_seeds)
    )

    completed_trials = 0
    result_rows = []

    for number_of_observations in observation_counts:
        for run_number, random_seed in enumerate(
            random_seeds,
            start=1,
        ):
            completed_trials += 1

            print(
                f"Running {completed_trials}/{total_trials}: "
                f"observations={number_of_observations:,}, "
                f"run={run_number}"
            )

            generated_data = generate_cdm_choice_data(
                interaction_matrix=true_interactions,
                number_of_observations=number_of_observations,
                choice_set_sizes=choice_set_sizes,
                random_seed=random_seed,
            )

            training_data, test_data = split_cdm_choice_data(
                data=generated_data,
                train_fraction=train_fraction,
                random_seed=random_seed,
            )

            optimization_result = optimize_full_cdm_with_adam(
                number_of_items=number_of_items,
                training_data=training_data,
                learning_rate=learning_rate,
                gradient_tolerance=gradient_tolerance,
                nll_tolerance=nll_tolerance,
                patience=patience,
                maximum_epochs=maximum_epochs,
                dtype=dtype,
            )

            parameter_rmse = cdm_parameter_recovery_error(
                true_interactions=true_interactions,
                estimated_interactions=(
                    optimization_result.interaction_matrix
                ),
            )

            off_diagonal_mask = ~torch.eye(
                number_of_items,
                device=true_interactions.device,
                dtype=torch.bool,
            )

            true_values = true_interactions[
                off_diagonal_mask
            ]

            estimated_values = (
                optimization_result.interaction_matrix[
                    off_diagonal_mask
                ]
            )

            parameter_correlation = torch.corrcoef(
                torch.stack(
                    [
                        true_values,
                        estimated_values,
                    ]
                )
            )[0, 1].item()

            with torch.no_grad():
                uniform_test_nll = torch.log(
                    test_data.choice_mask.sum(
                        dim=1
                    ).to(dtype)
                ).mean().item()

                true_test_nll = (
                    average_cdm_negative_log_likelihood(
                        interaction_matrix=true_interactions,
                        choice_sets=test_data.choice_sets,
                        chosen_positions=(
                            test_data.chosen_positions
                        ),
                        choice_mask=test_data.choice_mask,
                    ).item()
                )

                estimated_test_nll = (
                    average_cdm_negative_log_likelihood(
                        interaction_matrix=(
                            optimization_result.interaction_matrix
                        ),
                        choice_sets=test_data.choice_sets,
                        chosen_positions=(
                            test_data.chosen_positions
                        ),
                        choice_mask=test_data.choice_mask,
                    ).item()
                )

            result_rows.append(
                {
                    "Number of observations": (
                        number_of_observations
                    ),
                    "Run": run_number,
                    "Random seed": random_seed,
                    "Training observations": len(
                        training_data.choice_sets
                    ),
                    "Test observations": len(
                        test_data.choice_sets
                    ),
                    "Parameter RMSE": parameter_rmse,
                    "Parameter correlation": (
                        parameter_correlation
                    ),
                    "Uniform test NLL": uniform_test_nll,
                    "True test NLL": true_test_nll,
                    "Estimated test NLL": (
                        estimated_test_nll
                    ),
                    "Epochs": (
                        optimization_result.epochs_completed
                    ),
                    "Runtime (seconds)": (
                        optimization_result.runtime_seconds
                    ),
                    "Converged": (
                        optimization_result.converged
                    ),
                    "Stopping reason": (
                        optimization_result.stopping_reason
                    ),
                }
            )

    print("All observation-scaling trials completed.")

    return result_rows


def run_cdm_item_scaling_experiment(
    item_counts: Sequence[int],
    number_of_observations: int,
    choice_set_sizes: Sequence[int],
    random_seeds: Sequence[int],
    *,
    parameter_scale: float = 0.5,
    parameter_seed: int = 0,
    train_fraction: float = 0.8,
    learning_rate: float = 0.05,
    gradient_tolerance: float = 1e-5,
    nll_tolerance: float = 1e-9,
    patience: int = 50,
    maximum_epochs: int = 5_000,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> list[dict]:
    """Run repeated full-CDM experiments for different item counts."""

    experiment_results = []

    for number_of_items in item_counts:
        current_parameter_seed = (
            parameter_seed + number_of_items
        )

        true_interactions = generate_cdm_parameters(
            number_of_items=number_of_items,
            parameter_scale=parameter_scale,
            random_seed=current_parameter_seed,
            device=device,
            dtype=dtype,
        )

        current_results = (
            run_cdm_observation_scaling_experiment(
                true_interactions=true_interactions,
                observation_counts=[
                    number_of_observations
                ],
                choice_set_sizes=choice_set_sizes,
                random_seeds=random_seeds,
                train_fraction=train_fraction,
                learning_rate=learning_rate,
                gradient_tolerance=gradient_tolerance,
                nll_tolerance=nll_tolerance,
                patience=patience,
                maximum_epochs=maximum_epochs,
            )
        )

        number_of_free_parameters = (
            number_of_items
            * (number_of_items - 1)
            - 1
        )

        for result in current_results:
            result = dict(result)

            result["Number of items"] = (
                number_of_items
            )
            result["Number of free parameters"] = (
                number_of_free_parameters
            )
            result["Parameter seed"] = (
                current_parameter_seed
            )

            experiment_results.append(result)

    return experiment_results


def run_cdm_to_btl_experiment(
    true_interactions: torch.Tensor,
    observation_counts: Sequence[int],
    choice_set_sizes: Sequence[int],
    random_seeds: Sequence[int],
    *,
    train_fraction: float = 0.8,
    choix_alpha: float = 0.0,
    maximum_iterations: int = 1_000,
    tolerance: float = 1e-8,
) -> list[dict]:
    """Generate CDM data and fit a context-independent BTL/Luce model."""

    number_of_items = true_interactions.shape[0]

    number_of_cdm_parameters = (
        number_of_items
        * (number_of_items - 1)
        - 1
    )

    number_of_btl_parameters = (
        number_of_items - 1
    )

    total_trials = (
        len(observation_counts)
        * len(random_seeds)
    )

    completed_trials = 0
    result_rows = []

    for number_of_observations in observation_counts:
        for run_number, random_seed in enumerate(
            random_seeds,
            start=1,
        ):
            completed_trials += 1

            split_seed = random_seed + 1

            print(
                f"Running {completed_trials}/{total_trials}: "
                f"observations={number_of_observations:,}, "
                f"run={run_number}"
            )

            generated_data = generate_cdm_choice_data(
                interaction_matrix=true_interactions,
                number_of_observations=number_of_observations,
                choice_set_sizes=choice_set_sizes,
                random_seed=random_seed,
            )

            training_data, test_data = split_cdm_choice_data(
                data=generated_data,
                train_fraction=train_fraction,
                random_seed=split_seed,
            )

            training_observations = (
                _convert_cdm_data_to_choice_observations(
                    training_data
                )
            )

            test_observations = (
                _convert_cdm_data_to_choice_observations(
                    test_data
                )
            )

            fitting_start_time = perf_counter()

            estimated_btl_parameters = (
                estimate_top1_parameters_with_choix(
                    number_of_items=number_of_items,
                    observations=training_observations,
                    alpha=choix_alpha,
                    maximum_iterations=(
                        maximum_iterations
                    ),
                    tolerance=tolerance,
                )
            )

            btl_runtime_seconds = (
                perf_counter() - fitting_start_time
            )

            uniform_parameters = np.zeros(
                number_of_items,
                dtype=float,
            )

            uniform_training_nll = (
                average_top1_negative_log_likelihood(
                    observations=training_observations,
                    parameters=uniform_parameters,
                )
            )

            uniform_test_nll = (
                average_top1_negative_log_likelihood(
                    observations=test_observations,
                    parameters=uniform_parameters,
                )
            )

            btl_training_nll = (
                average_top1_negative_log_likelihood(
                    observations=training_observations,
                    parameters=estimated_btl_parameters,
                )
            )

            btl_test_nll = (
                average_top1_negative_log_likelihood(
                    observations=test_observations,
                    parameters=estimated_btl_parameters,
                )
            )

            with torch.no_grad():
                true_cdm_training_nll = (
                    average_cdm_negative_log_likelihood(
                        interaction_matrix=true_interactions,
                        choice_sets=(
                            training_data.choice_sets
                        ),
                        chosen_positions=(
                            training_data.chosen_positions
                        ),
                        choice_mask=(
                            training_data.choice_mask
                        ),
                    ).item()
                )

                true_cdm_test_nll = (
                    average_cdm_negative_log_likelihood(
                        interaction_matrix=true_interactions,
                        choice_sets=test_data.choice_sets,
                        chosen_positions=(
                            test_data.chosen_positions
                        ),
                        choice_mask=test_data.choice_mask,
                    ).item()
                )

            training_misspecification_gap = (
                btl_training_nll
                - true_cdm_training_nll
            )

            test_misspecification_gap = (
                btl_test_nll
                - true_cdm_test_nll
            )

            btl_training_improvement = (
                uniform_training_nll
                - btl_training_nll
            )

            btl_test_improvement = (
                uniform_test_nll
                - btl_test_nll
            )

            result_rows.append(
                {
                    "Number of items": number_of_items,
                    "Number of observations": (
                        number_of_observations
                    ),
                    "Run": run_number,
                    "Random seed": random_seed,
                    "Split seed": split_seed,
                    "Training observations": len(
                        training_observations
                    ),
                    "Test observations": len(
                        test_observations
                    ),
                    "Number of free CDM parameters": (
                        number_of_cdm_parameters
                    ),
                    "Number of free BTL parameters": (
                        number_of_btl_parameters
                    ),
                    "Uniform training NLL": (
                        uniform_training_nll
                    ),
                    "Uniform test NLL": (
                        uniform_test_nll
                    ),
                    "True CDM training NLL": (
                        true_cdm_training_nll
                    ),
                    "True CDM test NLL": (
                        true_cdm_test_nll
                    ),
                    "BTL/Luce training NLL": (
                        btl_training_nll
                    ),
                    "BTL/Luce test NLL": (
                        btl_test_nll
                    ),
                    "Training misspecification gap": (
                        training_misspecification_gap
                    ),
                    "Test misspecification gap": (
                        test_misspecification_gap
                    ),
                    "BTL training improvement over uniform": (
                        btl_training_improvement
                    ),
                    "BTL test improvement over uniform": (
                        btl_test_improvement
                    ),
                    "BTL runtime (seconds)": (
                        btl_runtime_seconds
                    ),
                } 
            )

    print("All CDM-to-BTL trials completed.")

    return result_rows
def run_cdm_to_btl_context_strength_experiment(
    base_interactions: torch.Tensor,
    interaction_scales: Sequence[float],
    number_of_observations: int,
    choice_set_sizes: Sequence[int],
    random_seeds: Sequence[int],
    *,
    train_fraction: float = 0.8,
    choix_alpha: float = 0.0,
    maximum_iterations: int = 1_000,
    tolerance: float = 1e-8,
) -> list[dict]:
    """Evaluate BTL/Luce misspecification across CDM interaction strengths."""

    if base_interactions.ndim != 2:
        raise ValueError(
            "base_interactions must be a matrix"
        )

    if (
        base_interactions.shape[0]
        != base_interactions.shape[1]
    ):
        raise ValueError(
            "base_interactions must be square"
        )

    if len(interaction_scales) == 0:
        raise ValueError(
            "interaction_scales must not be empty"
        )

    experiment_results = []

    for interaction_scale in interaction_scales:
        if interaction_scale < 0:
            raise ValueError(
                "interaction scales must be non-negative"
            )

        print(
            "\nInteraction scale: "
            f"{interaction_scale:.2f}"
        )

        scaled_interactions = (
            float(interaction_scale)
            * base_interactions
        )

        current_results = run_cdm_to_btl_experiment(
            true_interactions=scaled_interactions,
            observation_counts=[
                number_of_observations
            ],
            choice_set_sizes=choice_set_sizes,
            random_seeds=random_seeds,
            train_fraction=train_fraction,
            choix_alpha=choix_alpha,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
        )

        for result in current_results:
            result = dict(result)

            result["Interaction scale"] = (
                float(interaction_scale)
            )

            experiment_results.append(result)

    print(
        "All context-strength trials completed."
    )

    return experiment_results