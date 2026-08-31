"""Graph diagnostics for Bradley-Terry MLE analysis."""

import numpy as np


def _validate_number_of_items(number_of_items):
    """Validate the number of items."""
    if not isinstance(
        number_of_items,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_items must be an integer"
        )

    if number_of_items < 2:
        raise ValueError(
            "number_of_items must be at least two"
        )


def _as_comparison_array(
    comparisons,
    number_of_items,
):
    """Convert (winner, loser) observations into an array."""
    _validate_number_of_items(number_of_items)

    comparison_array = np.asarray(comparisons)

    if comparison_array.size == 0:
        return np.empty(
            (0, 2),
            dtype=int,
        )

    if (
        comparison_array.ndim != 2
        or comparison_array.shape[1] != 2
    ):
        raise ValueError(
            "comparisons must contain "
            "(winner, loser) pairs"
        )

    if not np.issubdtype(
        comparison_array.dtype,
        np.integer,
    ):
        raise TypeError(
            "comparisons must contain integer item indices"
        )

    comparison_array = comparison_array.astype(
        int,
        copy=False,
    )

    if (
        np.any(comparison_array < 0)
        or np.any(
            comparison_array >= number_of_items
        )
    ):
        raise ValueError(
            "comparisons contain an invalid item index"
        )

    if np.any(
        comparison_array[:, 0]
        == comparison_array[:, 1]
    ):
        raise ValueError(
            "an item cannot be compared with itself"
        )

    return comparison_array


def _find_reachable_items(
    starting_item,
    adjacency_list,
):
    """Find every item reachable from one starting item."""
    visited_items = set()
    items_to_visit = [starting_item]

    while items_to_visit:
        current_item = items_to_visit.pop()

        if current_item in visited_items:
            continue

        visited_items.add(current_item)

        items_to_visit.extend(
            adjacency_list[current_item]
        )

    return visited_items


def extract_undirected_comparison_edges(
    number_of_items,
    comparisons,
):
    """Return unique observed pairs while ignoring outcomes."""
    comparison_array = _as_comparison_array(
        comparisons,
        number_of_items,
    )

    edges = {
        tuple(sorted((winner, loser)))
        for winner, loser in comparison_array
    }

    return sorted(edges)


def extract_directed_win_edges(
    number_of_items,
    comparisons,
):
    """Return unique directed winner-to-loser edges."""
    comparison_array = _as_comparison_array(
        comparisons,
        number_of_items,
    )

    edges = {
        (int(winner), int(loser))
        for winner, loser in comparison_array
    }

    return sorted(edges)


def observed_comparison_graph_is_connected(
    number_of_items,
    comparisons,
):
    """Check connectivity after ignoring comparison outcomes."""
    comparison_edges = (
        extract_undirected_comparison_edges(
            number_of_items,
            comparisons,
        )
    )

    adjacency_list = {
        item: []
        for item in range(number_of_items)
    }

    for item_i, item_j in comparison_edges:
        adjacency_list[item_i].append(item_j)
        adjacency_list[item_j].append(item_i)

    reachable_items = _find_reachable_items(
        starting_item=0,
        adjacency_list=adjacency_list,
    )

    return (
        len(reachable_items)
        == number_of_items
    )


def directed_win_graph_is_strongly_connected(
    number_of_items,
    comparisons,
):
    """Check strong connectivity of the directed win graph."""
    directed_edges = extract_directed_win_edges(
        number_of_items,
        comparisons,
    )

    forward_adjacency = {
        item: []
        for item in range(number_of_items)
    }

    reverse_adjacency = {
        item: []
        for item in range(number_of_items)
    }

    for winner, loser in directed_edges:
        forward_adjacency[winner].append(loser)
        reverse_adjacency[loser].append(winner)

    forward_reachable = _find_reachable_items(
        starting_item=0,
        adjacency_list=forward_adjacency,
    )

    if len(forward_reachable) != number_of_items:
        return False

    reverse_reachable = _find_reachable_items(
        starting_item=0,
        adjacency_list=reverse_adjacency,
    )

    return (
        len(reverse_reachable)
        == number_of_items
    )


def ford_condition_holds(
    number_of_items,
    comparisons,
):
    """Check Ford's condition for a finite Bradley-Terry MLE."""
    return directed_win_graph_is_strongly_connected(
        number_of_items,
        comparisons,
    )