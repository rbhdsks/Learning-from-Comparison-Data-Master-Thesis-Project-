"""Public functions for the Bradley-Terry experiments."""

from .choix_estimation import (
    convert_choice_observations_to_choix_top1,
    estimate_pairwise_parameters_with_choix,
    estimate_top1_parameters_with_choix,
)
from .diagnostics import (
    directed_win_graph_is_strongly_connected,
    extract_directed_win_edges,
    extract_undirected_comparison_edges,
    ford_condition_holds,
    observed_comparison_graph_is_connected,
)
from .estimation import (
    estimate_parameters_with_gradient_descent,
    estimate_parameters_with_gradient_descent_trace,
)
from .experiments import (
    run_parameter_recovery_trials,
)
from .metrics import (
    average_top1_negative_log_likelihood,
    parameter_recovery_error,
)
from .model import (
    average_negative_log_likelihood,
    average_negative_log_likelihood_gradient,
    bradley_terry_probability,
    center_parameters,
    sigmoid,
)
from .optimization import (
    OptimizationResult,
    optimize_with_adam,
    optimize_with_full_batch_gradient_descent,
    optimize_with_minibatch_gradient_descent,
    optimize_with_momentum_gradient_descent,
    optimize_with_stochastic_gradient_descent,
)
from .pytorch_optimization import (
    optimize_with_pytorch_adam,
    optimize_with_pytorch_full_batch_gradient_descent,
    optimize_with_pytorch_minibatch_gradient_descent,
    optimize_with_pytorch_momentum_gradient_descent,
    optimize_with_pytorch_stochastic_gradient_descent,
)
from .sampling import (
    build_complete_comparison_edges,
    build_connected_erdos_renyi_edges,
    build_cycle_comparison_edges,
    build_star_comparison_edges,
    build_two_clusters_with_bridge_edges,
    build_two_disconnected_clusters_edges,
    comparison_graph_is_connected,
)
from .simulation import (
    generate_comparison_data_from_edges,
    generate_uniform_comparison_data,
)


__all__ = [
    "OptimizationResult",
    "average_negative_log_likelihood",
    "average_negative_log_likelihood_gradient",
    "average_top1_negative_log_likelihood",
    "bradley_terry_probability",
    "build_complete_comparison_edges",
    "build_connected_erdos_renyi_edges",
    "build_cycle_comparison_edges",
    "build_star_comparison_edges",
    "build_two_clusters_with_bridge_edges",
    "build_two_disconnected_clusters_edges",
    "center_parameters",
    "comparison_graph_is_connected",
    "convert_choice_observations_to_choix_top1",
    "directed_win_graph_is_strongly_connected",
    "estimate_pairwise_parameters_with_choix",
    "estimate_parameters_with_gradient_descent",
    "estimate_parameters_with_gradient_descent_trace",
    "estimate_top1_parameters_with_choix",
    "extract_directed_win_edges",
    "extract_undirected_comparison_edges",
    "ford_condition_holds",
    "generate_comparison_data_from_edges",
    "generate_uniform_comparison_data",
    "observed_comparison_graph_is_connected",
    "optimize_with_adam",
    "optimize_with_full_batch_gradient_descent",
    "optimize_with_minibatch_gradient_descent",
    "optimize_with_momentum_gradient_descent",
    "optimize_with_pytorch_adam",
    "optimize_with_pytorch_full_batch_gradient_descent",
    "optimize_with_pytorch_minibatch_gradient_descent",
    "optimize_with_pytorch_momentum_gradient_descent",
    "optimize_with_pytorch_stochastic_gradient_descent",
    "optimize_with_stochastic_gradient_descent",
    "parameter_recovery_error",
    "run_parameter_recovery_trials",
    "sigmoid",
]