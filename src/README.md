# Source Code

This directory contains the reusable code used by the experiment notebooks.
Notebooks define settings, run experiments, and interpret results; model,
simulation, estimation, and evaluation logic lives in `src/`.

## Bradley–Terry package

The `bradley_terry/` package contains the pairwise-comparison workflow.
It includes implementations used in the foundational experiments and
estimation through the Choix library.

| File | Purpose |
| --- | --- |
| `bradley_terry/model.py` | Bradley–Terry probabilities, likelihood, and gradients |
| `bradley_terry/sampling.py` | Sampling comparisons between items |
| `bradley_terry/simulation.py` | Generating synthetic comparison data |
| `bradley_terry/estimation.py` | Estimating Bradley–Terry parameters |
| `bradley_terry/choix_estimation.py` | Estimating parameters with Choix |
| `bradley_terry/optimization.py` | Reusable numerical optimization routines |
| `bradley_terry/pytorch_optimization.py` | PyTorch-based optimization routines |
| `bradley_terry/metrics.py` | Parameter-recovery and evaluation metrics |
| `bradley_terry/diagnostics.py` | Checks and diagnostics for fitted models |
| `bradley_terry/experiments.py` | Reusable routines for repeated experiments |
| `bradley_terry/__init__.py` | Package-level imports for notebook use |

## Context-Dependent Model package

The `context_dependent_model/` package supports synthetic choice
experiments with full and low-rank CDM implementations in PyTorch.

| File | Purpose |
| --- | --- |
| `context_dependent_model/model.py` | CDM choice probabilities and full and low-rank model classes |
| `context_dependent_model/sampling.py` | Choice-set sampling utilities |
| `context_dependent_model/simulation.py` | Generating CDM parameters and synthetic choices |
| `context_dependent_model/estimation.py` | Fitting full and low-rank CDM models |
| `context_dependent_model/metrics.py` | Parameter-recovery and evaluation metrics |
| `context_dependent_model/experiments.py` | Reusable CDM experiment routines |
| `context_dependent_model/__init__.py` | Package-level imports for notebook use |

## Working convention

Keep reusable functions in these packages. A notebook should contain the
experiment-specific configuration, calls to those functions, plots,
result logging, and an interpretation of what the results show.