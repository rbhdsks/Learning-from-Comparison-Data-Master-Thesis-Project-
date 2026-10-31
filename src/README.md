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

The interaction and probability functions are also used by the cyclic-preference
foundation notebook. A reusable general cycle detector is planned in the
cycle-detection design notebook and is not yet part of this package.

## CDM paper-reproduction modules

The `cdm_paper_reproduction/` directory contains routines used by the
synthetic recovery and IIA-testing notebooks. These build on the project's
BTL and CDM implementations; the original paper code remains under
`references/paper_code/cdm_icml/`.

| File | Purpose |
| --- | --- |
| `cdm_paper_reproduction/synthetic.py` | Paper-style parameter generation and uniform sampling over subset menus |
| `cdm_paper_reproduction/estimation.py` | Full-CDM Adam fitting with optional initial interactions |
| `cdm_paper_reproduction/experiments.py` | Repeated recovery trials, checkpointing, and zero-choice pair diagnostics |
| `cdm_paper_reproduction/iia_simulation.py` | Menu enumeration, general choice probabilities, and synthetic choices |
| `cdm_paper_reproduction/iia_metrics.py` | Universal choice-model fitting from empirical menu-wise frequencies |
| `cdm_paper_reproduction/iia_testing.py` | MNL-versus-CDM and MNL-versus-universal likelihood-ratio tests with complete menu coverage |
| `cdm_paper_reproduction/iia_scaling.py` | Observed-menu model dimensions, fits, and IIA-test diagnostics |
| `cdm_paper_reproduction/iia_separation.py` | Linear-programming diagnostic for separating directions in full-CDM data |

An optimizer's numerical convergence flag does not establish that a finite
unregularized MLE exists. Interpret fitted likelihoods and nominal
chi-square tests alongside separation, menu coverage, boundary counts,
and stopping diagnostics.

## Working convention

Keep reusable functions in these packages. A notebook should contain the
experiment-specific configuration, calls to those functions, plots,
result logging, and an interpretation of what the results show.

Saved numerical outputs belong in `results/experiments/`. Thesis figures
and tables belong in `docs/thesis/images/experiments/` and
`docs/thesis/tables/experiments/`, respectively. See the
[notebook guide](../notebooks/README.md) for the experiments using these modules.
