# Source Code

This directory contains the reusable implementations used by the experiment notebooks, while the notebooks define experimental settings, is displaying results and producing plots. The modules inside `src/` contain the underlying model, simulation, estimation, optimization and evaluation codes.

## Package structure

| Path | Purpose |
| --- | --- |
| `bradley_terry/model.py` | Bradley–Terry probabilities, negative log-likelihood and gradient calculations |
| `bradley_terry/simulation.py` | Generation of simulated pairwise-comparison data |
| `bradley_terry/estimation.py` | Parameter estimation algorithms |
| `bradley_terry/metrics.py` | Evaluation measures such as parameter-recovery error |
| `bradley_terry/experiments.py` | Utilities for running repeated experiments |
| `bradley_terry/__init__.py` | Public functions exposed by the package |


