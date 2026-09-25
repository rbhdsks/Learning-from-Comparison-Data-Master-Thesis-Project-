"""Reusable PyTorch tools for Context-Dependent Models."""

from .estimation import (
    CDMOptimizationResult,
    LowRankCDMOptimizationResult,
    optimize_full_cdm_with_adam,
    optimize_low_rank_cdm_with_adam,
)
from .experiments import (
    run_cdm_item_scaling_experiment,
    run_cdm_observation_scaling_experiment,
    run_cdm_to_btl_context_strength_experiment,
    run_cdm_to_btl_experiment,
)
from .metrics import (
    cdm_parameter_recovery_error,
)
from .model import (
    FullContextDependentModel,
    LowRankContextDependentModel,
    average_cdm_negative_log_likelihood,
    cdm_choice_probabilities,
    cdm_choice_utilities,
    center_cdm_parameters,
)
from .simulation import (
    CDMChoiceData,
    generate_cdm_choice_data,
    generate_cdm_choice_data_on_sets,
    generate_cdm_parameters,
    generate_uniform_choice_sets,
    split_cdm_choice_data,
)

__all__ = [
    "CDMChoiceData",
    "CDMOptimizationResult",
    "FullContextDependentModel",
    "LowRankCDMOptimizationResult",
    "LowRankContextDependentModel",
    "average_cdm_negative_log_likelihood",
    "cdm_choice_probabilities",
    "cdm_choice_utilities",
    "cdm_parameter_recovery_error",
    "center_cdm_parameters",
    "generate_cdm_choice_data",
    "generate_cdm_choice_data_on_sets",
    "generate_cdm_parameters",
    "generate_uniform_choice_sets",
    "optimize_full_cdm_with_adam",
    "optimize_low_rank_cdm_with_adam",
    "run_cdm_item_scaling_experiment",
    "run_cdm_observation_scaling_experiment",
    "run_cdm_to_btl_context_strength_experiment",
    "run_cdm_to_btl_experiment",
    "split_cdm_choice_data",
]