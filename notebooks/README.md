# Notebooks

This directory contains the mathematical foundations, numerical
experiments, and archived exploratory work developed during the project.

Reusable implementations are maintained inside `src/`. The notebooks
primarily define experiment configurations, run the reusable functions,
and interpret the resulting tables and figures.

## Directory structure

| Path | Purpose |
| --- | --- |
| `foundations/` | Mathematical derivations and small numerical examples |
| `experiments/` | Main thesis experiments |
| `archive/` | Earlier exploratory notebooks retained for reference |

## Foundations

### Bradley–Terry Model and MLE

The notebooks under `foundations/Bradley_Terry_Model_MLE/` introduce the
ideas in the following order:

| Notebook | Topic |
| --- | --- |
| `00-mle.ipynb` | Introduction to maximum-likelihood estimation |
| `01-simple_mle.ipynb` | Small MLE example |
| `02-log-likelihood.ipynb` | Likelihood and log-likelihood |
| `03-general_bernoulli_likelihood.ipynb` | Bernoulli likelihood |
| `04-MLE-NLL-GradientDescent.ipynb` | Negative log-likelihood and gradient descent |
| `05-two-item-bradley-terry.ipynb` | Two-item Bradley–Terry model |
| `06-three-item-bradley-terry-recovery.ipynb` | Three-item parameter recovery |

Additional foundation notebooks study Bradley–Terry optimization and the
Context-Dependent Model using small examples.

## Main experiments

### Bradley–Terry Model

| Notebook | Experiment |
| --- | --- |
| `01_BradleyTerry_recovery.ipynb` | Parameter recovery and the effect of the number of comparisons |
| `02_BradleyTerry_sampling.ipynb` | Comparison of pair-sampling procedures |
| `03_BradleyTerry_MLE_existence_uniqueness.ipynb` | Ford's condition and existence of a finite MLE |
| `04_BradleyTerry_MLE_Optimizations.ipynb` | Comparison of handcrafted and PyTorch optimizers |

### Context-Dependent Model

| Notebook | Experiment |
| --- | --- |
| `01_CDM.ipynb` | Initial implementation and validation of the Context-Dependent Model |

The Context-Dependent Model experiments are currently under development.

## Notebook conventions

- Notebooks are numbered in their intended reading order.
- Each experiment uses explicit configuration values and random seeds.
- The term *parameter* is used consistently for model parameters.
- Reusable functions should be moved to `src/`.
- Generated notebook outputs are stored in the corresponding `outputs/`
  directory.
- Figures selected for the thesis are stored under
  `docs/thesis/images/`.
- Notebooks should be restarted and run from beginning to end before
  being committed.