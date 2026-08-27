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


Functions can then be imported directly inside notebooks in this particular manner :

```python
from bradley_terry import (
    estimate_parameters_with_gradient_descent,
    generate_uniform_comparison_data,
    parameter_recovery_error,
)
```

The project is installed in editable mode. Changes made inside `src/` are therefore available without reinstalling the package, although the notebook kernel may need to be restarted after changing an imported module.

## Adding new models

Also the future models will be added as separate packages inside `src/`:

```text
src/
├── bradley_terry/
├── cdm/
└── other_models/
```

Each package also contain an `__init__.py` file and separate modules for its main responsibilities.


