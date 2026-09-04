"""PyTorch optimization methods for the Bradley-Terry model."""

from collections.abc import Sequence
from time import perf_counter
from typing import Literal

import numpy as np
import torch

from .optimization import OptimizationResult


Comparison = tuple[int, int]


def _prepare_comparisons(
    number_of_items: int,
    comparisons: Sequence[Comparison],
) -> torch.Tensor:
    """Validate comparisons and convert them into a tensor."""

    comparison_array = np.asarray(
        comparisons,
        dtype=np.int64,
    )

    if (
        comparison_array.ndim != 2
        or comparison_array.shape[1] != 2
    ):
        raise ValueError(
            "comparisons must contain (winner, loser) pairs"
        )

    if len(comparison_array) == 0:
        raise ValueError("comparisons cannot be empty")

    if (
        comparison_array.min() < 0
        or comparison_array.max() >= number_of_items
    ):
        raise ValueError(
            "comparisons contain an invalid item index"
        )

    if np.any(
        comparison_array[:, 0]
        == comparison_array[:, 1]
    ):
        raise ValueError(
            "winner and loser must be different items"
        )

    return torch.tensor(
        comparison_array,
        dtype=torch.long,
        device="cpu",
    )


def _prepare_initial_parameters(
    number_of_items: int,
    initial_parameters: np.ndarray | None,
) -> torch.Tensor:
    """Create centred float64 PyTorch parameters."""

    if initial_parameters is None:
        parameter_array = np.zeros(
            number_of_items,
            dtype=float,
        )
    else:
        parameter_array = np.asarray(
            initial_parameters,
            dtype=float,
        ).copy()

        if parameter_array.shape != (number_of_items,):
            raise ValueError(
                "initial_parameters has an invalid shape"
            )

        parameter_array -= parameter_array.mean()

    return torch.tensor(
        parameter_array,
        dtype=torch.float64,
        device="cpu",
        requires_grad=True,
    )


def _average_pytorch_nll(
    parameters: torch.Tensor,
    comparisons: torch.Tensor,
) -> torch.Tensor:
    """Calculate stable average Bradley-Terry NLL."""

    winners = comparisons[:, 0]
    losers = comparisons[:, 1]

    parameter_differences = (
        parameters[losers]
        - parameters[winners]
    )

    return torch.nn.functional.softplus(
        parameter_differences
    ).mean()


def _evaluate_full_dataset(
    parameters: torch.Tensor,
    comparisons: torch.Tensor,
) -> tuple[float, float]:
    """Calculate full-data NLL and full-gradient norm."""

    full_nll = _average_pytorch_nll(
        parameters,
        comparisons,
    )

    full_gradient = torch.autograd.grad(
        full_nll,
        parameters,
    )[0]

    return (
        float(full_nll.detach().item()),
        float(
            torch.linalg.vector_norm(
                full_gradient
            ).detach().item()
        ),
    )


def _find_stopping_reason(
    current_nll: float,
    gradient_norm: float,
    gradient_tolerance: float,
    target_nll: float | None,
) -> str | None:
    """Check whether a numerical stopping condition holds."""

    if gradient_norm <= gradient_tolerance:
        return "gradient tolerance reached"

    if (
        target_nll is not None
        and current_nll <= target_nll
    ):
        return "target NLL reached"

    return None


def _create_optimizer(
    optimizer_kind: Literal[
        "sgd",
        "momentum",
        "adam",
    ],
    parameters: torch.Tensor,
    learning_rate: float,
    momentum: float,
    beta_1: float,
    beta_2: float,
    epsilon: float,
) -> torch.optim.Optimizer:
    """Construct the requested PyTorch optimizer."""

    if optimizer_kind == "sgd":
        return torch.optim.SGD(
            [parameters],
            lr=learning_rate,
        )

    if optimizer_kind == "momentum":
        return torch.optim.SGD(
            [parameters],
            lr=learning_rate,
            momentum=momentum,
        )

    if optimizer_kind == "adam":
        return torch.optim.Adam(
            [parameters],
            lr=learning_rate,
            betas=(beta_1, beta_2),
            eps=epsilon,
        )

    raise ValueError(
        f"unknown optimizer kind: {optimizer_kind}"
    )


def _run_pytorch_optimizer(
    *,
    optimizer_name: str,
    optimizer_kind: Literal[
        "sgd",
        "momentum",
        "adam",
    ],
    number_of_items: int,
    comparisons: Sequence[Comparison],
    learning_rate: float,
    batch_size: int,
    shuffle: bool,
    random_seed: int,
    gradient_tolerance: float,
    target_nll: float | None,
    initial_parameters: np.ndarray | None,
    emergency_maximum_epochs: int,
    momentum: float = 0.9,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
) -> OptimizationResult:
    """Run one PyTorch optimizer until convergence."""

    if number_of_items < 2:
        raise ValueError(
            "number_of_items must be at least two"
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive"
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive"
        )

    if gradient_tolerance <= 0:
        raise ValueError(
            "gradient_tolerance must be positive"
        )

    if emergency_maximum_epochs < 1:
        raise ValueError(
            "emergency_maximum_epochs must be positive"
        )

    comparison_tensor = _prepare_comparisons(
        number_of_items,
        comparisons,
    )

    parameters = _prepare_initial_parameters(
        number_of_items,
        initial_parameters,
    )

    number_of_observations = len(
        comparison_tensor
    )

    batch_size = min(
        batch_size,
        number_of_observations,
    )

    optimizer = _create_optimizer(
        optimizer_kind=optimizer_kind,
        parameters=parameters,
        learning_rate=learning_rate,
        momentum=momentum,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )

    random_generator = torch.Generator(
        device="cpu"
    )

    random_generator.manual_seed(
        random_seed
    )

    initial_nll, initial_gradient_norm = (
        _evaluate_full_dataset(
            parameters,
            comparison_tensor,
        )
    )

    nll_history = [initial_nll]
    gradient_norm_history = [
        initial_gradient_norm
    ]

    parameter_history = [
        parameters.detach().numpy().copy()
    ]

    update_count_history = [0]

    number_of_updates = 0
    number_of_epochs = 0

    stopping_reason = _find_stopping_reason(
        current_nll=initial_nll,
        gradient_norm=initial_gradient_norm,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
    )

    start_time = perf_counter()

    while (
        stopping_reason is None
        and number_of_epochs
        < emergency_maximum_epochs
    ):
        if shuffle:
            observation_indices = torch.randperm(
                number_of_observations,
                generator=random_generator,
            )
        else:
            observation_indices = torch.arange(
                number_of_observations
            )

        for start_index in range(
            0,
            number_of_observations,
            batch_size,
        ):
            end_index = (
                start_index + batch_size
            )

            batch_indices = observation_indices[
                start_index:end_index
            ]

            comparison_batch = comparison_tensor[
                batch_indices
            ]

            optimizer.zero_grad()

            batch_nll = _average_pytorch_nll(
                parameters,
                comparison_batch,
            )

            batch_nll.backward()
            optimizer.step()

            # Preserve the Bradley-Terry sum-to-zero constraint.
            with torch.no_grad():
                parameters -= parameters.mean()

            number_of_updates += 1

        number_of_epochs += 1

        current_nll, gradient_norm = (
            _evaluate_full_dataset(
                parameters,
                comparison_tensor,
            )
        )

        nll_history.append(current_nll)
        gradient_norm_history.append(
            gradient_norm
        )

        parameter_history.append(
            parameters.detach().numpy().copy()
        )

        update_count_history.append(
            number_of_updates
        )

        stopping_reason = _find_stopping_reason(
            current_nll=current_nll,
            gradient_norm=gradient_norm,
            gradient_tolerance=gradient_tolerance,
            target_nll=target_nll,
        )

    runtime_seconds = (
        perf_counter() - start_time
    )

    if stopping_reason is None:
        stopping_reason = (
            "emergency maximum epochs reached"
        )

    converged = (
        stopping_reason
        != "emergency maximum epochs reached"
    )

    return OptimizationResult(
        optimizer_name=optimizer_name,
        parameters=(
            parameters.detach().numpy().copy()
        ),
        nll_history=np.asarray(
            nll_history,
            dtype=float,
        ),
        gradient_norm_history=np.asarray(
            gradient_norm_history,
            dtype=float,
        ),
        parameter_history=np.asarray(
            parameter_history,
            dtype=float,
        ),
        update_count_history=np.asarray(
            update_count_history,
            dtype=int,
        ),
        runtime_seconds=runtime_seconds,
        converged=converged,
        stopping_reason=stopping_reason,
    )


def optimize_with_pytorch_full_batch_gradient_descent(
    number_of_items: int,
    comparisons: Sequence[Comparison],
    *,
    learning_rate: float = 0.1,
    gradient_tolerance: float = 1e-6,
    target_nll: float | None = None,
    initial_parameters: np.ndarray | None = None,
    emergency_maximum_epochs: int = 100_000,
) -> OptimizationResult:
    """Run PyTorch full-batch gradient descent."""

    return _run_pytorch_optimizer(
        optimizer_name="PyTorch full-batch GD",
        optimizer_kind="sgd",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=len(comparisons),
        shuffle=False,
        random_seed=0,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        initial_parameters=initial_parameters,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )


def optimize_with_pytorch_stochastic_gradient_descent(
    number_of_items: int,
    comparisons: Sequence[Comparison],
    *,
    learning_rate: float = 0.01,
    random_seed: int = 0,
    gradient_tolerance: float = 1e-6,
    target_nll: float | None = None,
    initial_parameters: np.ndarray | None = None,
    emergency_maximum_epochs: int = 10_000,
) -> OptimizationResult:
    """Run PyTorch stochastic gradient descent."""

    return _run_pytorch_optimizer(
        optimizer_name="PyTorch SGD",
        optimizer_kind="sgd",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=1,
        shuffle=True,
        random_seed=random_seed,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        initial_parameters=initial_parameters,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )


def optimize_with_pytorch_minibatch_gradient_descent(
    number_of_items: int,
    comparisons: Sequence[Comparison],
    *,
    learning_rate: float = 0.05,
    batch_size: int = 128,
    random_seed: int = 0,
    gradient_tolerance: float = 1e-6,
    target_nll: float | None = None,
    initial_parameters: np.ndarray | None = None,
    emergency_maximum_epochs: int = 10_000,
) -> OptimizationResult:
    """Run PyTorch mini-batch gradient descent."""

    return _run_pytorch_optimizer(
        optimizer_name="PyTorch mini-batch SGD",
        optimizer_kind="sgd",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        initial_parameters=initial_parameters,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
    )


def optimize_with_pytorch_momentum_gradient_descent(
    number_of_items: int,
    comparisons: Sequence[Comparison],
    *,
    learning_rate: float = 0.02,
    batch_size: int = 128,
    momentum: float = 0.9,
    random_seed: int = 0,
    gradient_tolerance: float = 1e-6,
    target_nll: float | None = None,
    initial_parameters: np.ndarray | None = None,
    emergency_maximum_epochs: int = 10_000,
) -> OptimizationResult:
    """Run PyTorch mini-batch SGD with momentum."""

    return _run_pytorch_optimizer(
        optimizer_name="PyTorch momentum SGD",
        optimizer_kind="momentum",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        initial_parameters=initial_parameters,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
        momentum=momentum,
    )


def optimize_with_pytorch_adam(
    number_of_items: int,
    comparisons: Sequence[Comparison],
    *,
    learning_rate: float = 0.01,
    batch_size: int = 128,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-8,
    random_seed: int = 0,
    gradient_tolerance: float = 1e-6,
    target_nll: float | None = None,
    initial_parameters: np.ndarray | None = None,
    emergency_maximum_epochs: int = 10_000,
) -> OptimizationResult:
    """Run the PyTorch Adam optimizer."""

    return _run_pytorch_optimizer(
        optimizer_name="PyTorch Adam",
        optimizer_kind="adam",
        number_of_items=number_of_items,
        comparisons=comparisons,
        learning_rate=learning_rate,
        batch_size=batch_size,
        shuffle=True,
        random_seed=random_seed,
        gradient_tolerance=gradient_tolerance,
        target_nll=target_nll,
        initial_parameters=initial_parameters,
        emergency_maximum_epochs=(
            emergency_maximum_epochs
        ),
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
    )