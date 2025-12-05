import jax
import jax.numpy as jnp
import numpy as np
import time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from optimization.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from optimization.utils import t2j_numpy, j2t_numpy


class BotorchBO(OptimizationAlgorithm):
    """Bayesian Optimization using BoTorch (PyTorch-based, GPU-accelerated)."""

    algorithm_str: str = "botorch_bo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self, problem: ContinuousProblem, max_iterations: int = 100) -> None:
        self.problem = problem
        self.max_iterations = max_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        init_params: Float[Array, "{self.problem.n_params}"] | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_times: list[int | float] | None = None,
        lb: Float[Array, "{self.problem.n_params}"] | None = None,
        ub: Float[Array, "{self.problem.n_params}"] | None = None,
        n_initial: int = 5,
        **bo_kwargs,
    ) -> tuple[
        Float[Array, "{self.problem.n_params}"],
        Float[Array, "n_iters {self.problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Optimize the given problem using BoTorch Bayesian Optimization.

        Args:
            save_to_file (bool): Whether to save results to file.
            init_params (Float[Array, "{self.problem.n_params}"] | None): Initial parameters.
            return_best_params_history (bool): Whether to return history of best parameters.
            random_seed (int | None): Random seed for reproducibility.
            wall_times (list[int | float] | None): Wall time checkpoints.
            lb (Float[Array, "{self.problem.n_params}"] | None): Lower bounds for parameters.
            ub (Float[Array, "{self.problem.n_params}"] | None): Upper bounds for parameters.
            n_initial (int): Number of initial random samples before fitting GP.
            **bo_kwargs: Additional keyword arguments.

        Returns:
            tuple: Best parameters, history of best parameters, losses, wall time indices.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Set up bounds
        if lb is None:
            lb = jnp.full(self.problem.n_params, -10.0)
        if ub is None:
            ub = jnp.full(self.problem.n_params, 10.0)

        bounds = torch.tensor(
            np.array([lb, ub]), device=self.device, dtype=self.dtype
        )

        # Normalization helpers: map between original space and unit cube [0, 1]^d
        bounds_range = bounds[1] - bounds[0]
        def normalize(x: torch.Tensor) -> torch.Tensor:
            return (x - bounds[0]) / bounds_range

        def unnormalize(x: torch.Tensor) -> torch.Tensor:
            return x * bounds_range + bounds[0]

        # Unit cube bounds for GP
        unit_bounds = torch.tensor(
            [[0.0] * self.problem.n_params, [1.0] * self.problem.n_params],
            device=self.device,
            dtype=self.dtype,
        )

        # JIT-compile objective function
        @jax.jit
        def jax_objective(params):
            return self.problem.objective_function(params)

        # Wrapper to convert torch -> jax -> torch (operates in original space)
        def evaluate(x: torch.Tensor) -> torch.Tensor:
            y = float(jax_objective(t2j_numpy(x)))
            return torch.tensor([y], device=self.device, dtype=self.dtype)

        # Warmup JIT
        _ = jax_objective(jnp.zeros(self.problem.n_params))

        # Initialize variables
        best_params = jnp.zeros(self.problem.n_params)
        best_params_history = []
        best_loss = float("inf")
        losses = []

        # Initialize wall_time tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        max_wall_time: float | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        # Generate initial samples (stored in normalized [0,1] space)
        if init_params is not None:
            train_X_orig = torch.tensor(
                np.array(init_params).reshape(1, -1),
                device=self.device,
                dtype=self.dtype,
            )
            train_X = normalize(train_X_orig)
            train_Y = evaluate(train_X_orig[0]).unsqueeze(0)
        else:
            train_X = torch.empty(0, self.problem.n_params, device=self.device, dtype=self.dtype)
            train_Y = torch.empty(0, 1, device=self.device, dtype=self.dtype)

        # Add random initial points (in normalized space)
        n_random = max(0, n_initial - len(train_X))
        if n_random > 0:
            random_X = torch.rand(
                n_random, self.problem.n_params, device=self.device, dtype=self.dtype
            )
            random_Y = torch.stack([evaluate(unnormalize(x)) for x in random_X])
            train_X = torch.cat([train_X, random_X], dim=0) if len(train_X) > 0 else random_X
            train_Y = torch.cat([train_Y, random_Y], dim=0) if len(train_Y) > 0 else random_Y

        # Record initial losses
        for y in train_Y:
            loss = float(y.item())
            losses.append(loss)
            if loss < best_loss - 1e-4:
                best_loss = loss
                idx = len(losses) - 1
                best_params = t2j_numpy(unnormalize(train_X[idx]))

            if return_best_params_history:
                best_params_history.append(best_params)

        def run_iteration(i: int, train_X: torch.Tensor, train_Y: torch.Tensor):
            """Run a single BO iteration. Returns updated train_X, train_Y, and loss.
            
            train_X is in normalized [0,1] space.
            """
            # Fit GP model (minimize loss, so negate Y for EI which maximizes)
            gp = SingleTaskGP(train_X, -train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Optimize acquisition function in unit cube
            acq = LogExpectedImprovement(gp, best_f=-best_loss)
            candidate, _ = optimize_acqf(
                acq,
                bounds=unit_bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )

            # Evaluate candidate in original space
            candidate_orig = unnormalize(candidate[0])
            new_Y = evaluate(candidate_orig)
            train_X = torch.cat([train_X, candidate], dim=0)
            train_Y = torch.cat([train_Y, new_Y.unsqueeze(0)], dim=0)

            loss = float(new_Y.item())
            return train_X, train_Y, loss, candidate_orig

        # Main optimization loop
        if wall_times is not None:
            start_time = time.time()
            i = len(losses)

            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time

                # Record iteration index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(i)
                    wall_times_remaining.popleft()

                train_X, train_Y, loss, candidate = run_iteration(i, train_X, train_Y)

                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss = loss
                    best_params = t2j_numpy(candidate)
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(loss)
                i += 1

            # Fill remaining wall_times that weren't reached
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()

        else:
            for i in range(len(losses), self.max_iterations):
                train_X, train_Y, loss, candidate = run_iteration(i, train_X, train_Y)

                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss = loss
                    best_params = t2j_numpy(candidate)
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(loss)

        losses = jnp.array(losses)
        best_params_history = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        if save_to_file:
            self.problem.output_to_files(
                best_params=best_params,
                losses=losses,
                population_losses=None,
                algorithm_str=self.algorithm_str,
                hyper_param_str="",
                hyper_param_str_in_filename=False,
            )

        return best_params, best_params_history, losses, wall_time_indices
