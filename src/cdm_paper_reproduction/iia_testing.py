"""Reusable likelihood-ratio tests for complete-menu choice datasets."""

import numpy as np
import torch
from scipy.stats import chi2

from bradley_terry.choix_estimation import (
    estimate_top1_parameters_with_choix,
)
from context_dependent_model.model import (
    average_cdm_negative_log_likelihood,
    center_cdm_parameters,
)
from .estimation import optimize_full_cdm_with_adam
from .iia_metrics import fit_universal_choice_model


def run_complete_menu_iia_tests(
    data,
    catalogue_choice_sets,
    catalogue_choice_mask,
    number_of_items,
    *,
    significance_level=0.05,
    maximum_epochs=5_000,
    dtype=torch.float64,
) -> dict:
    """Fit three choice models and calculate two asymptotic IIA tests."""

    universal_fit = fit_universal_choice_model(
        data=data,
        catalogue_choice_sets=catalogue_choice_sets,
        catalogue_choice_mask=catalogue_choice_mask,
    )

    if universal_fit["unobserved_menus"] != 0:
        raise ValueError(
            "This runner requires every catalogue menu to be observed."
        )

    observations = [
        (
            int(chosen_item),
            tuple(int(item) for item in items[mask].tolist()),
        )
        for chosen_item, items, mask in zip(
            data.chosen_items,
            data.choice_sets,
            data.choice_mask,
        )
    ]

    estimated_parameters = estimate_top1_parameters_with_choix(
        number_of_items=number_of_items,
        observations=observations,
        alpha=0.0,
        maximum_iterations=10_000,
        tolerance=1e-10,
    )

    parameters = torch.as_tensor(
        estimated_parameters,
        dtype=dtype,
        device=data.choice_sets.device,
    )

    with torch.no_grad():
        utilities = parameters[
            data.choice_sets.clamp_min(0)
        ].masked_fill(~data.choice_mask, -torch.inf)

        chosen_utilities = utilities.gather(
            1,
            data.chosen_positions.unsqueeze(1),
        ).squeeze(1)

        mnl_nll = (
            torch.logsumexp(utilities, dim=1)
            - chosen_utilities
        ).mean().item()

    initial_interactions = center_cdm_parameters(
        -parameters.unsqueeze(0)
        .expand(number_of_items, -1)
        .clone()
    )

    with torch.no_grad():
        initial_nll = average_cdm_negative_log_likelihood(
            interaction_matrix=initial_interactions,
            choice_sets=data.choice_sets,
            chosen_positions=data.chosen_positions,
            choice_mask=data.choice_mask,
        ).item()

    if not np.isclose(initial_nll, mnl_nll, atol=1e-10, rtol=0):
        raise ValueError("CDM initialization does not reproduce MNL NLL.")

    result = optimize_full_cdm_with_adam(
        number_of_items=number_of_items,
        training_data=data,
        initial_interactions=initial_interactions,
        learning_rate=0.05,
        gradient_tolerance=1e-5,
        nll_tolerance=1e-9,
        patience=50,
        maximum_epochs=maximum_epochs,
        dtype=dtype,
    )

    with torch.no_grad():
        cdm_nll = average_cdm_negative_log_likelihood(
            interaction_matrix=result.interaction_matrix,
            choice_sets=data.choice_sets,
            chosen_positions=data.chosen_positions,
            choice_mask=data.choice_mask,
        ).item()

    universal_nll = universal_fit["mean_nll"]
    number_of_observations = len(observations)

    if not np.isfinite([mnl_nll, cdm_nll, universal_nll]).all():
        raise ValueError("A fitted NLL is not finite.")

    if universal_nll > cdm_nll + 1e-8:
        raise ValueError("Universal NLL exceeds fitted CDM NLL.")

    mnl_dimension = number_of_items - 1
    cdm_dimension = number_of_items * (number_of_items - 1) - 1
    universal_dimension = int(
        (catalogue_choice_mask.sum(dim=1) - 1).sum().item()
    )

    output = {
        "Observations": number_of_observations,
        "MNL mean NLL": mnl_nll,
        "CDM mean NLL": cdm_nll,
        "Universal mean NLL": universal_nll,
        "CDM converged": bool(result.converged),
        "CDM epochs": result.epochs_completed,
        "CDM runtime (seconds)": result.runtime_seconds,
        "CDM stopping reason": result.stopping_reason,
        "Observed menus": universal_fit["observed_menus"],
        "Zero-count available outcomes": (
            universal_fit["zero_count_available_cells"]
        ),
    }

    alternatives = [
        ("CDM", cdm_nll, cdm_dimension, bool(result.converged)),
        ("Universal", universal_nll, universal_dimension, True),
    ]

    for name, alternative_nll, dimension, converged in alternatives:
        raw_statistic = (
            2 * number_of_observations * (mnl_nll - alternative_nll)
        )

        numerically_usable = converged and raw_statistic >= -1e-7
        degrees_of_freedom = dimension - mnl_dimension

        p_value = (
            float(chi2.sf(max(raw_statistic, 0.0), degrees_of_freedom))
            if numerically_usable
            else np.nan
        )

        output[f"{name} LR statistic"] = raw_statistic
        output[f"{name} degrees of freedom"] = degrees_of_freedom
        output[f"{name} test computed"] = numerically_usable
        output[f"{name} p-value"] = p_value
        output[f"{name} reject MNL"] = (
            bool(p_value < significance_level)
            if numerically_usable
            else None
        )

    return output