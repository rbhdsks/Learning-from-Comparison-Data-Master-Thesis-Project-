# Learning from Comparison Data

Research code and [thesis notes](https://github.com/rbhdsks/Learning-from-Comparison-Data-Master-Thesis-Project-/blob/main/docs/thesis/Notes.pdf) for my M.Tech project on learning preference models from observed choice data.

The project begins with the Bradley–Terry model for pairwise comparisons and extends to full and low-rank context-dependent choice models, synthetic paper reproduction, and the study of cyclic preferences.

## Research workflow

```mermaid
flowchart LR
    A["Foundations<br/>Probability, MLE and gradient descent"]
    B["Phase 0<br/>Bradley–Terry model"]
    C["Parameter recovery<br/>and data scaling"]
    D["MLE existence<br/>and Ford's condition"]
    E["Optimization<br/>Handcrafted and PyTorch"]
    F["Phase 1 — Ongoing<br/>Context-Dependent Model"]
    G["Full CDM<br/>Synthetic experiments"]
    H["Cross-model fits<br/>BTL and CDM"]
    I["Unified experiment<br/>BTL, full and low-rank CDM"]
    J["CDM paper reproduction<br/>Recovery and IIA pilots"]
    K["Computational scaling<br/>Full and low-rank CDM"]
    L["Cyclic preferences<br/>Mathematical examples verified"]
    M["Cycle detection<br/>Design and exact checks; simulations planned"]

    A --> B
    B --> C
    B --> D
    B --> E
    C --> F
    D --> F
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    I --> K
    F --> L
    L --> M

    classDef foundation fill:#f3f4f6,stroke:#6b7280,color:#111827
    classDef completed fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef experiment fill:#eff6ff,stroke:#60a5fa,color:#111827
    classDef ongoing fill:#fef3c7,stroke:#d97706,color:#111827
    classDef planned fill:#f3f4f6,stroke:#9ca3af,color:#374151

    class A,L foundation
    class B completed
    class C,D,E experiment
    class F ongoing
    class G,H,K experiment
    class I,J ongoing
    class M planned
```

## Research progress

| Phase | Focus | Status |
| --- | --- | --- |
| Phase 0 | Bradley–Terry model and maximum-likelihood estimation | Core experiments completed |
| Phase 1 | Full and low-rank CDM, synthetic comparisons, computational scaling, and paper reproduction | Synthetic results and paper-reproduction pilots available; broader validation ongoing |
| Phase 1: Cyclic preferences | BTL–CDM relationships, cycle construction, and cycle-detection design | Mathematical examples verified; noisy-data detection experiments planned |

### Phase 0: Bradley–Terry Model and MLE

The completed experiments study:

- parameter recovery from simulated comparisons;
- recovery error as the number of comparisons and items changes;
- existence and uniqueness of the finite MLE, including Ford's condition;
- failure of the finite MLE under perfect separation;
- convergence and runtime of handcrafted and PyTorch optimizers.

### Representative Bradley–Terry results

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/assets/readme/btl_parameter_recovery.png"
           alt="Bradley–Terry parameter recovery" width="420">
      <br>
      <sub>True and estimated Bradley–Terry parameters.</sub>
    </td>
    <td align="center" width="50%">
      <img src="docs/assets/readme/btl_recovery_scaling.png"
           alt="Recovery error versus comparisons" width="420">
      <br>
      <sub>Parameter-recovery error with increasing comparisons.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/assets/readme/btl_nonfinite_mle.png"
           alt="Parameter growth without a finite MLE" width="420">
      <br>
      <sub>Parameter divergence when Ford's condition fails.</sub>
    </td>
    <td align="center" width="50%">
      <img src="docs/assets/readme/btl_optimizer_convergence.png"
           alt="Optimizer convergence comparison" width="420">
      <br>
      <sub>Final gradient norms relative to the convergence tolerance.</sub>
    </td>
  </tr>
</table>

### Phase 1: Context-Dependent Models

This phase covers full-CDM estimation, BTL–CDM cross-model experiments, low-rank CDM, computational scaling, and reproduction of experiments from the CDM paper.

**Unified synthetic experiment:** BTL, full CDM, and low-rank CDM are fitted to common synthetic choice sets and compared using held-out NLL, optimization diagnostics, and runtime.

#### Current synthetic results

<img src="docs/thesis/images/experiments/context_dependent_model/unified/n10_m10000_seed20260831_scale0.4/04_cross_fit_nll_gap.png"
     alt="Held-out NLL gaps for BTL and CDM cross-model fits" width="780">

*Held-out NLL cost of cross-model fitting on BTL- and CDM-generated choices.*

<img src="docs/thesis/images/experiments/context_dependent_model/unified/n10_m10000_seed20260831_scale0.4/04_low_rank_paired_test_nll_gaps.png"
     alt="Paired test NLL gaps for selected low-rank CDM fits" width="780">

*Paired test NLL gaps for the validation-selected low-rank CDM fits.*

These figures show one synthetic run. The selected rank-5 fit reached the epoch limit without meeting the gradient convergence criterion, so broader conclusions require further runs and optimization checks.

#### Full versus low-rank computational scaling

The resource-scaling notebook measures runtime, time per training step, trainable parameter count, and sampled process memory as the number of items increases. Models share observations within each repeat and use the same training-step budget.

These results benchmark the existing dense-matrix implementations. Sampled process memory includes Python and PyTorch overhead; the experiment measures computational cost under a fixed step budget, rather than time to convergence or parameter recovery.

#### CDM paper reproduction

The paper-reproduction notebooks now contain synthetic full-CDM parameter recovery and likelihood-ratio tests of Independence of Irrelevant Alternatives (IIA):

- **Synthetic recovery:** study recovery error as the number of observed choices increases, with repeated runs and diagnostics for rare choices and separation.
- **Synthetic IIA tests:** generate choices from MNL, full CDM, and a general choice system; compare MNL against full CDM and the universal choice model using repeated-dataset pilots.
- **Separation diagnostics:** distinguish numerical optimizer stopping from the existence of a finite unregularized CDM maximum-likelihood estimate.

The IIA results are pilot experiments. Incomplete fits, separation, and boundary outcomes require care when interpreting nominal chi-square likelihood-ratio tests. Larger repeated experiments and recovery extensions beyond the six-item paper setup remain further work.

#### Cyclic preferences and cycle detection

The foundation notebook [07 — BTL and CDM interactions and cycles](notebooks/foundations/Bradley_Terry_Model_MLE/07_BTL_vs_CDM_Interactions_and_Cycles.ipynb) verifies the CDM utility equations, constructs a three-item majority-preference cycle, demonstrates pairwise non-identifiability, and recovers BTL as a special case of CDM.

It establishes model relationships using specified parameters; it does not estimate cycles from noisy observations.

The [cycle-detection design notebook](notebooks/experiments/Context_Dependent_Model/Cyclic_Preferences/01_Cycle_Detection_Design.ipynb) contains executed exact-probability checks and a plan for graph detection, sampled comparisons, simultaneous confidence intervals, and repeated experiments. The general detector and noisy-data experiments are not yet implemented.

## Repository structure

| Path | Purpose |
| --- | --- |
| `notebooks/archive/` | Earlier notebooks preserved for reference |
| `notebooks/foundations/` | Mathematical derivations and small examples |
| `notebooks/archive/choix/` | Archived practice with the Choix library |
| `notebooks/experiments/` | Main thesis experiments |
| `src/bradley_terry/` | Reusable Bradley–Terry code |
| `src/context_dependent_model/` | Reusable full and low-rank CDM code |
| `src/cdm_paper_reproduction/` | Synthetic recovery, IIA tests, and separation diagnostics |
| `results/experiments/` | Saved numerical results, settings, and fit diagnostics |
| `docs/thesis/images/` | Thesis figures, including unified CDM results |
| `docs/thesis/tables/` | Tables generated for thesis experiments |
| `references/` | Reference implementations and supporting material |

See the [notebook guide](notebooks/README.md), [source-code guide](src/README.md), and [documentation guide](docs/README.md) for the detailed layout.

Code under `references/` belongs to its original authors and is kept separate from the implementation developed in this project.

## Setup

The project requires Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebooks,models]"
```

When opening a notebook, select `.venv` as the Python kernel.

## References

- Arjun Seshadri, Alex Peysakhovich, and Johan Ugander. [Discovering Context Effects from Raw Choice Data](https://proceedings.mlr.press/v97/seshadri19a.html). ICML, 2019.
- [Choix](https://github.com/lucasmaystre/choix): inference tools for models based on Luce's choice axiom.

## Project information

| | |
| --- | --- |
| Student | Nitesh Kumar Shah |
| Roll number | ID25M806 |
| Programme | M.Tech in Data Science and Artificial Intelligence |
| Supervisor | Prof. Suryanarayana Sankagiri |
| Institute | Indian Institute of Technology Madras |

This repository is under active development. Implementations, experiments, and documentation may change as the research progresses.
