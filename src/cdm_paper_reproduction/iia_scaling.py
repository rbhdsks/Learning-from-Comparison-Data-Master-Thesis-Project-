"""IIA testing helpers that support incomplete menu coverage."""

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


def get_observed_model_dimensions(
    catalogue_choice_sets,
    catalogue_choice_mask,
    menu_counts,
    number_of_items,
) -> dict:
    """Calculate model dimensions from observed-menu utility contrasts."""

    catalogue = catalogue_choice_sets.detach().cpu().numpy()
    masks = catalogue_choice_mask.detach().cpu().numpy().astype(bool)

    mnl_contrasts = []
    cdm_contrasts = []
    universal_dimension = 0

    for row_index in np.flatnonzero(np.asarray(menu_counts) > 0):
        items = catalogue[row_index][masks[row_index]].tolist()
        reference_item = items[0]

        universal_dimension += len(items) - 1

        for item in items[1:]:
            mnl_row = np.zeros(number_of_items)
            mnl_row[item] = 1.0
            mnl_row[reference_item] = -1.0
            mnl_contrasts.append(mnl_row)

            cdm_row = np.zeros((number_of_items, number_of_items))

            for context_item in items:
                if context_item != item:
                    cdm_row[item, context_item] += 1.0

                if context_item != reference_item:
                    cdm_row[reference_item, context_item] -= 1.0

            cdm_contrasts.append(cdm_row.ravel())

    if not mnl_contrasts:
        raise ValueError("No usable observed menus.")

    return {
        "MNL": int(np.linalg.matrix_rank(np.asarray(mnl_contrasts))),
        "CDM": int(np.linalg.matrix_rank(np.asarray(cdm_contrasts))),
        "Universal": int(universal_dimension),
    }


def run_observed_menu_iia_tests(
    data,
    catalogue_choice_sets,
    catalogue_choice_mask,
    number_of_items,
    *,
    significance_level=0.05,
    maximum_epochs=5_000,
    dtype=torch.float64,
) -> dict:
    """Fit choice models and retain diagnostics for both IIA tests."""

    universal_fit = fit_universal_choice_model(
        data=data,
        catalogue_choice_sets=catalogue_choice_sets,
        catalogue_choice_mask=catalogue_choice_mask,
    )

    dimensions = get_observed_model_dimensions(
        catalogue_choice_sets=catalogue_choice_sets,
        catalogue_choice_mask=catalogue_choice_mask,
        menu_counts=universal_fit["menu_counts"],
        number_of_items=number_of_items,
    )

    observations = [
        (
            int(chosen),
            tuple(int(item) for item in items[mask].tolist()),
        )
        for chosen, items, mask in zip(
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
            torch.logsumexp(utilities, dim=1) - chosen_utilities
        ).mean().item()

    universal_nll = universal_fit["mean_nll"]
    number_of_observations = len(observations)

    if not np.isfinite([mnl_nll, universal_nll]).all():
        raise ValueError("MNL or universal NLL is not finite.")

    if universal_nll > mnl_nll + 1e-8:
        raise ValueError("Universal NLL exceeds fitted MNL NLL.")

    output = {
        "MNL mean NLL": mnl_nll,
        "Universal mean NLL": universal_nll,
        "Observed menus": universal_fit["observed_menus"],
        "Unobserved menus": universal_fit["unobserved_menus"],
        "Zero-count available outcomes": (
            universal_fit["zero_count_available_cells"]
        ),
        "MNL dimension": dimensions["MNL"],
        "CDM dimension": dimensions["CDM"],
        "Universal dimension": dimensions["Universal"],
        "CDM mean NLL": np.nan,
        "CDM converged": False,
        "CDM epochs": np.nan,
        "CDM runtime (seconds)": np.nan,
        "CDM stopping reason": "",
        "CDM fit error": "",
    }

    try:
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
            raise ValueError("CDM initialization does not match MNL.")

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

        output.update({
            "CDM mean NLL": cdm_nll,
            "CDM converged": bool(result.converged),
            "CDM epochs": result.epochs_completed,
            "CDM runtime (seconds)": result.runtime_seconds,
            "CDM stopping reason": result.stopping_reason,
        })

    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        output["CDM fit error"] = f"{type(error).__name__}: {error}"

    for alternative in ["CDM", "Universal"]:
        alternative_nll = output[f"{alternative} mean NLL"]
        degrees_of_freedom = (
            dimensions[alternative] - dimensions["MNL"]
        )

        raw_statistic = (
            2 * number_of_observations * (mnl_nll - alternative_nll)
        )

        reason = ""

        if degrees_of_freedom <= 0:
            reason = "No additional identifiable dimensions"
        elif not np.isfinite(raw_statistic):
            reason = "Non-finite likelihood-ratio statistic"
        elif raw_statistic < -1e-7:
            reason = "Alternative likelihood below MNL likelihood"
        elif alternative == "CDM":
            if not output["CDM converged"]:
                reason = "CDM optimizer did not converge"
            elif alternative_nll < universal_nll - 1e-8:
                reason = "CDM likelihood exceeds universal likelihood"

        p_value = (
            float(chi2.sf(max(raw_statistic, 0.0), degrees_of_freedom))
            if not reason
            else np.nan
        )

        output[f"{alternative} LR statistic"] = raw_statistic
        output[f"{alternative} degrees of freedom"] = degrees_of_freedom
        output[f"{alternative} p-value"] = p_value
        output[f"{alternative} test computed"] = not bool(reason)
        output[f"{alternative} test issue"] = reason

    return output