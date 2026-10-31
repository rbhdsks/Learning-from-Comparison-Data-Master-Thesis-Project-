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
| `07_BTL_vs_CDM_Interactions_and_Cycles.ipynb` | CDM interactions, cyclic preferences, pairwise non-identifiability, and BTL as a special case of CDM |

Additional foundation notebooks study Bradley–Terry optimization and the
Context-Dependent Model using small examples.

Notebook 07 constructs and verifies a cyclic preference example from
specified parameters. It does not fit a model or implement a general
cycle detector for noisy observations.

## Main experiments

### Bradley–Terry Model

| Notebook | Experiment |
| --- | --- |
| `00_BTL_recovery_noofitems.ipynb` | Parameter recovery as the number of items changes |
| `01_BradleyTerry_recovery.ipynb` | Parameter recovery and the effect of the number of comparisons |
| `02_BradleyTerry_sampling.ipynb` | Comparison of pair-sampling procedures |
| `03_BradleyTerry_MLE_existence_uniqueness.ipynb` | Ford's condition and existence of a finite MLE |
| `04_BradleyTerry_MLE_Optimizations.ipynb` | Comparison of handcrafted and PyTorch optimizers |

### Full Context-Dependent Model

Directory: `experiments/Context_Dependent_Model/Full CDM/`.

| Notebook | Experiment |
| --- | --- |
| `01_CDM-CDM.ipynb` | Generate choices with CDM, fit CDM, and study parameter recovery |
| `02_CDM-BTL.ipynb` | Generate choices with CDM and fit BTL |
| `03-BTL-CDM.ipynb` | Generate choices with BTL and fit CDM |

These notebooks examine recovery and cross-model fitting in individual
settings. The unified notebook below compares the models under one setup.

### Full and low-rank CDM

Directory: `experiments/Context_Dependent_Model/Low Rank CDM/`.

| Notebook | Experiment |
| --- | --- |
| `01-FullvsLowrankResourceScaling.ipynb` | Runtime, time per step, parameter count, and sampled process memory under a shared training-step budget |
| `04-Unifiednotebook.ipynb` | BTL, full CDM, and low-rank CDM on shared synthetic choice sets using held-out NLL, optimization diagnostics, and runtime |

The resource benchmark uses the existing dense-matrix implementations;
it does not measure time to convergence. The unified results include a
selected rank-5 fit that reached the epoch limit without satisfying the
gradient convergence criterion. Broader conclusions require additional
runs and optimization checks.

### CDM paper reproduction

Directory: `experiments/Context_Dependent_Model/CDM_Paper_Reproduction/`.

| Notebook | Experiment |
| --- | --- |
| `01_synthetic_recovery.ipynb` | Six-item full-CDM parameter recovery across observation counts, repeated runs, and rare-choice/separation diagnostics |
| `02_synthetic_iia_tests.ipynb` | MNL-versus-CDM and MNL-versus-universal likelihood-ratio tests on MNL-, CDM-, and general-choice-generated data |

Both notebooks have saved executions and numerical results. The IIA
notebook includes repeated-dataset and observation-scaling pilots,
observed-menu dimension calculations, and linear-programming separation
diagnostics. These pilots do not establish precise finite-sample
calibration of the nominal chi-square tests. Recovery extensions beyond
six items and larger repeated studies remain further work.

### Cyclic preferences and cycle-detection design

Directory: `experiments/Context_Dependent_Model/Cyclic_Preferences/`.

| Notebook | Experiment |
| --- | --- |
| `01_Cycle_Detection_Design.ipynb` | Design draft with executed exact-probability checks for BTL, a CDM cycle, transitive non-BTL probabilities, and ties |

The draft builds on foundation notebook 07. It distinguishes observed-win
cycles, majority-preference cycles, and log-odds inconsistency. General
graph detection, noisy comparisons, simultaneous confidence intervals,
and repeated detection-power experiments are specified but not yet
implemented. Low-rank fitting and larger-menu context tests are later
extensions.

## Experiment outputs

Numerical results and configuration files are stored under
`results/experiments/` at the project root. Figures and tables selected for
the thesis are stored under `docs/thesis/images/experiments/` and
`docs/thesis/tables/experiments/`. Keep pilot results, exact examples,
and planned experiments clearly distinguished in interpretations.

## Notebook conventions

- Notebooks are numbered in their intended reading order.
- Each experiment uses explicit configuration values and random seeds.
- The term *parameter* is used consistently for model parameters.
- Reusable functions should be moved to `src/`.
- Generated data, settings, diagnostics, and runtimes should be logged with their experiment.
- Figures selected for the thesis are stored under `docs/thesis/images/`.
- Notebooks should be restarted and run from beginning to end before being committed.
