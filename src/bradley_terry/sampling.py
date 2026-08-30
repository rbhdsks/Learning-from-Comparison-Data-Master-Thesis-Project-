"""Comparison-graph construction for Bradley-Terry experiments."""

import numpy as np


def _validate_number_of_items(
    number_of_items,
    minimum=2,
):
    """Validate the number of items."""
    if not isinstance(
        number_of_items,
        (int, np.integer),
    ):
        raise TypeError(
            "number_of_items must be an integer"
        )

    if number_of_items < minimum:
        raise ValueError(
            f"number_of_items must be at least {minimum}"
        )


def build_complete_comparison_edges(number_of_items):
    """Return every possible unordered pair of items."""
    _validate_number_of_items(number_of_items)

    return [
        (item_i, item_j)
        for item_i in range(number_of_items)
        for item_j in range(
            item_i + 1,
            number_of_items,
        )
    ]


def build_cycle_comparison_edges(number_of_items):
    """Return edges forming a connected cycle."""
    _validate_number_of_items(
        number_of_items,
        minimum=3,
    )

    edges = {
        tuple(sorted((
            item,
            (item + 1) % number_of_items,
        )))
        for item in range(number_of_items)
    }

    return sorted(edges)


def build_star_comparison_edges(
    number_of_items,
    central_item=0,
):
    """Return edges connecting one central item to all others."""
    _validate_number_of_items(number_of_items)

    if not isinstance(
        central_item,
        (int, np.integer),
    ):
        raise TypeError(
            "central_item must be an integer"
        )

    if not 0 <= central_item < number_of_items:
        raise ValueError(
            "central_item must be a valid item index"
        )

    return [
        tuple(sorted((central_item, item)))
        for item in range(number_of_items)
        if item != central_item
    ]


def build_two_disconnected_clusters_edges(
    number_of_items,
):
    """Return two complete clusters with no connecting edge."""
    _validate_number_of_items(
        number_of_items,
        minimum=4,
    )

    split_index = number_of_items // 2

    first_cluster = range(
        0,
        split_index,
    )

    second_cluster = range(
        split_index,
        number_of_items,
    )

    first_cluster_edges = [
        (item_i, item_j)
        for item_i in first_cluster
        for item_j in first_cluster
        if item_i < item_j
    ]

    second_cluster_edges = [
        (item_i, item_j)
        for item_i in second_cluster
        for item_j in second_cluster
        if item_i < item_j
    ]

    return (
        first_cluster_edges
        + second_cluster_edges
    )


def build_two_clusters_with_bridge_edges(
    number_of_items,
):
    """Return two complete clusters joined by one edge."""
    disconnected_edges = (
        build_two_disconnected_clusters_edges(
            number_of_items
        )
    )

    split_index = number_of_items // 2

    bridge_edge = (
        split_index - 1,
        split_index,
    )

    return disconnected_edges + [bridge_edge]


def comparison_graph_is_connected(
    number_of_items,
    edges,
):
    """Check whether every item is reachable in the graph."""
    _validate_number_of_items(number_of_items)

    edge_array = np.asarray(edges)

    if edge_array.size == 0:
        return False

    if (
        edge_array.ndim != 2
        or edge_array.shape[1] != 2
    ):
        raise ValueError(
            "edges must contain pairs of item indices"
        )

    if not np.issubdtype(
        edge_array.dtype,
        np.integer,
    ):
        raise TypeError(
            "edges must contain integer item indices"
        )

    if (
        np.any(edge_array < 0)
        or np.any(edge_array >= number_of_items)
    ):
        raise ValueError(
            "edges contain an invalid item index"
        )

    if np.any(
        edge_array[:, 0] == edge_array[:, 1]
    ):
        raise ValueError(
            "an item cannot be connected to itself"
        )

    neighbours = {
        item: []
        for item in range(number_of_items)
    }

    for item_i, item_j in edge_array:
        neighbours[item_i].append(item_j)
        neighbours[item_j].append(item_i)

    visited_items = set()
    items_to_visit = [0]

    while items_to_visit:
        current_item = items_to_visit.pop()

        if current_item in visited_items:
            continue

        visited_items.add(current_item)

        items_to_visit.extend(
            neighbours[current_item]
        )

    return (
        len(visited_items)
        == number_of_items
    )


def build_connected_erdos_renyi_edges(
    number_of_items,
    edge_probability,
    rng,
    maximum_attempts=10_000,
):
    """Sample an Erdős-Rényi graph conditional on connectivity."""
    _validate_number_of_items(number_of_items)

    if (
        not np.isfinite(edge_probability)
        or not 0 < edge_probability <= 1
    ):
        raise ValueError(
            "edge_probability must lie in (0, 1]"
        )

    if not isinstance(rng, np.random.Generator):
        raise TypeError(
            "rng must be created using np.random.default_rng()"
        )

    if not isinstance(
        maximum_attempts,
        (int, np.integer),
    ):
        raise TypeError(
            "maximum_attempts must be an integer"
        )

    if maximum_attempts <= 0:
        raise ValueError(
            "maximum_attempts must be positive"
        )

    possible_edges = (
        build_complete_comparison_edges(
            number_of_items
        )
    )

    for _ in range(maximum_attempts):
        sampled_edges = [
            edge
            for edge in possible_edges
            if rng.random() < edge_probability
        ]

        if comparison_graph_is_connected(
            number_of_items,
            sampled_edges,
        ):
            return sampled_edges

    raise RuntimeError(
        "could not generate a connected graph; "
        "increase edge_probability"
    )