"""Simulation of pairwise-comparison data."""

import numpy as np

from .model import bradley_terry_probability


def generate_uniform_comparison_data(
    true_parameters,
    number_of_comparisons,
    rng
):
    """Generate comparisons by sampling every item uniformly."""
    parameter_array=np.asarray(
        true_parameters,
        dtype=float
    )

    if parameter_array.ndim!=1:
        raise ValueError(
            "true_parameters must be a one-dimensional array"
        )

    if parameter_array.size<2:
        raise ValueError(
            "at least two true parameters are required"
        )

    if not np.all(np.isfinite(parameter_array)):
        raise ValueError(
            "true_parameters must contain only finite values"
        )

    if not isinstance(
        number_of_comparisons,
        (int,np.integer)
    ):
        raise TypeError(
            "number_of_comparisons must be an integer"
        )

    if number_of_comparisons<=0:
        raise ValueError(
            "number_of_comparisons must be positive"
        )

    if not isinstance(rng,np.random.Generator):
        raise TypeError(
            "rng must be created using np.random.default_rng()"
        )

    number_of_items=len(parameter_array)

    comparisons=[]

    for _ in range(number_of_comparisons):
        item_i,item_j=rng.choice(
            number_of_items,
            size=2,
            replace=False
        )

        probability_i_wins=bradley_terry_probability(
            item_i,
            item_j,
            parameter_array
        )

        if rng.random()<probability_i_wins:
            winner,loser=item_i,item_j
        else:
            winner,loser=item_j,item_i

        comparisons.append(
            (int(winner),int(loser))
        )

    return comparisons
def build_complete_comparison_edges(number_of_items):
    """Return every possible unordered pair of items."""
    return [
        (item_i, item_j)
        for item_i in range(number_of_items)
        for item_j in range(item_i + 1, number_of_items)
    ]


def build_cycle_comparison_edges(number_of_items):
    """Return edges forming a connected cycle."""
    if number_of_items < 3:
        raise ValueError(
            "a cycle requires at least three items"
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
    if not 0 <= central_item < number_of_items:
        raise ValueError(
            "central_item must be a valid item index"
        )

    return [
        tuple(sorted((central_item, item)))
        for item in range(number_of_items)
        if item != central_item
    ]


def build_two_clusters_with_bridge_edges(number_of_items):
    """Return two complete clusters connected by one bridge."""
    if number_of_items < 4:
        raise ValueError(
            "two clusters require at least four items"
        )

    split_index = number_of_items // 2

    first_cluster = range(0, split_index)
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

    bridge_edge = (
        split_index - 1,
        split_index,
    )

    return (
        first_cluster_edges
        + second_cluster_edges
        + [bridge_edge]
    )


def comparison_graph_is_connected(
    number_of_items,
    edges,
):
    """Check whether every item is reachable in the graph."""
    neighbours = {
        item: []
        for item in range(number_of_items)
    }

    for item_i, item_j in edges:
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

    return len(visited_items) == number_of_items


def build_connected_erdos_renyi_edges(
    number_of_items,
    edge_probability,
    rng,
):
    """Sample an Erdős–Rényi graph until it is connected."""
    if not 0 < edge_probability <= 1:
        raise ValueError(
            "edge_probability must lie in (0, 1]"
        )

    possible_edges = (
        build_complete_comparison_edges(
            number_of_items
        )
    )

    while True:
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