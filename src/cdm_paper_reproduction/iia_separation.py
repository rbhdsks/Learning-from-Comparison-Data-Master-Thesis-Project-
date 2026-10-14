# Diagnose separating directions in full-CDM choice data.

import numpy as np
from scipy.optimize import linprog


def diagnose_cdm_separation(
    data,
    number_of_items: int,
    *,
    tolerance: float = 1e-7,
) -> dict:
    """Search for a likelihood-improving direction using linear programming."""

    choice_sets = data.choice_sets.detach().cpu().numpy()
    choice_mask = data.choice_mask.detach().cpu().numpy().astype(bool)
    chosen_positions = data.chosen_positions.detach().cpu().numpy()

    contrast_rows = []

    for items, mask, position in zip(
        choice_sets,
        choice_mask,
        chosen_positions,
    ):
        available_items = items[mask]
        winner = int(items[int(position)])

        for loser in available_items:
            loser = int(loser)

            if loser == winner:
                continue

            contrast = np.zeros(
                (number_of_items, number_of_items),
                dtype=float,
            )

            for context in available_items:
                context = int(context)

                if context != winner:
                    contrast[winner, context] += 1.0

                if context != loser:
                    contrast[loser, context] -= 1.0

            contrast_rows.append(contrast.ravel())

    if not contrast_rows:
        raise ValueError("No nontrivial choice comparisons were found.")

    contrasts = np.unique(
        np.asarray(contrast_rows),
        axis=0,
    )

    direction_bounds = [
        (0.0, 0.0) if row == column else (-1.0, 1.0)
        for row in range(number_of_items)
        for column in range(number_of_items)
    ]

    solution = linprog(
        c=-contrasts.mean(axis=0),
        A_ub=-contrasts,
        b_ub=np.zeros(len(contrasts)),
        bounds=direction_bounds,
        method="highs",
    )

    if not solution.success:
        raise RuntimeError(
            f"Separation diagnostic failed: {solution.message}"
        )

    margins = contrasts @ solution.x

    separation_detected = bool(
        margins.min() >= -tolerance
        and margins.mean() > tolerance
    )

    return {
        "Separation detected": separation_detected,
        "Unique contrast constraints": len(contrasts),
        "Minimum directional margin": float(margins.min()),
        "Maximum directional margin": float(margins.max()),
        "Mean directional margin": float(margins.mean()),
        "Strictly improved contrasts": int(
            np.count_nonzero(margins > tolerance)
        ),
        "Direction matrix": solution.x.reshape(
            number_of_items,
            number_of_items,
        ),
        "Solver message": solution.message,
    }