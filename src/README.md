# Source Code

This directory contains the reusable implementations used by the
experiment notebooks. Notebooks define the experimental settings and
present the results, while model and estimation logic is maintained
inside `src/`.

## Packages

| Path | Purpose |
| --- | --- |
| `bradley_terry/model.py` | Bradley–Terry probabilities, negative log-likelihood, and gradients |
| `bradley_terry/simulation.py` | Simulation of pairwise-comparison data |
| `bradley_terry/estimation.py` | Parameter-estimation algorithms |
| `bradley_terry/metrics.py` | Parameter-recovery and evaluation metrics |
| `bradley_terry/experiments.py` | Utilities for repeated experiments |
| `bradley_terry/choix_estimation.py` | Estimation using the Choix library |
| `context_dependent_model/model.py` | PyTorch implementation of the Context-Dependent Model |

Reusable logic should remain inside `src/`; notebooks should primarily
configure, execute, and interpret experiments.