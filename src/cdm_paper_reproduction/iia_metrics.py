"""Reusable likelihood calculations for synthetic IIA experiments."""

import numpy as np


def fit_universal_choice_model(
    data,
    catalogue_choice_sets,
    catalogue_choice_mask,
) -> dict:
    """Estimate separate choice probabilities for each observed menu."""

    catalogue = catalogue_choice_sets.detach().cpu().numpy()
    mask = catalogue_choice_mask.detach().cpu().numpy().astype(bool)

    if catalogue.shape != mask.shape:
        raise ValueError("Catalogue and mask must have matching shapes.")

    menu_lookup = {}
    item_position_lookup = {}

    for row_index, (items, valid) in enumerate(zip(catalogue, mask)):
        available_items = items[valid].tolist()
        menu = tuple(sorted(available_items))

        if len(menu) < 2 or len(set(menu)) != len(menu):
            raise ValueError("Each menu must contain distinct items.")

        if menu in menu_lookup:
            raise ValueError("The catalogue contains duplicate menus.")

        menu_lookup[menu] = row_index
        item_position_lookup[menu] = {
            int(items[position]): int(position)
            for position in np.flatnonzero(valid)
        }

    counts = np.zeros(catalogue.shape, dtype=np.int64)

    observed_sets = data.choice_sets.detach().cpu().numpy()
    observed_masks = data.choice_mask.detach().cpu().numpy().astype(bool)
    chosen_items = data.chosen_items.detach().cpu().numpy()

    number_of_observations = len(chosen_items)

    if number_of_observations == 0:
        raise ValueError("Cannot fit an empty dataset.")

    for items, valid, chosen_item in zip(
        observed_sets,
        observed_masks,
        chosen_items,
    ):
        menu = tuple(sorted(items[valid].tolist()))

        if menu not in menu_lookup:
            raise ValueError(f"Observed menu absent from catalogue: {menu}")

        if int(chosen_item) not in item_position_lookup[menu]:
            raise ValueError("Chosen item is absent from its menu.")

        row_index = menu_lookup[menu]
        column_index = item_position_lookup[menu][int(chosen_item)]
        counts[row_index, column_index] += 1

    menu_counts = counts.sum(axis=1)
    observed_menu_mask = menu_counts > 0

    probabilities = np.zeros(counts.shape, dtype=np.float64)
    probabilities[observed_menu_mask] = (
        counts[observed_menu_mask]
        / menu_counts[observed_menu_mask, None]
    )
    probabilities[~observed_menu_mask & mask.any(axis=1)] = np.nan
    probabilities[~mask] = 0.0

    positive_counts = counts > 0

    total_log_likelihood = float(
        np.sum(
            counts[positive_counts]
            * np.log(probabilities[positive_counts])
        )
    )

    return {
        "choice_counts": counts,
        "menu_counts": menu_counts,
        "probabilities": probabilities,
        "number_of_observations": number_of_observations,
        "observed_menus": int(observed_menu_mask.sum()),
        "unobserved_menus": int((~observed_menu_mask).sum()),
        "zero_count_available_cells": int(
            (
                mask
                & observed_menu_mask[:, None]
                & (counts == 0)
            ).sum()
        ),
        "total_log_likelihood": total_log_likelihood,
        "mean_nll": -total_log_likelihood / number_of_observations,
    }