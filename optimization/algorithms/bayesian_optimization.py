import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from bayes_opt import BayesianOptimization as BO
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from optimization.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)


class BayesianOptimization(OptimizationAlgorithm):
    algorithm_str: str = "bayesian_optimization"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self, problem: ContinuousProblem, max_iterations: int = 100) -> None:
        self.problem = problem
        self.max_iterations = max_iterations
        # Additional initialization for Bayesian Optimization can be added here

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
        **bo_kwargs,
    ) -> tuple[
        Float[Array, "{self.problem.n_params}"],
        Float[Array, "n_iters {self.problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Optimize the given problem using Bayesian Optimization.

        Args:
            save_to_file (bool): Whether to save results to file.
            init_params (Float[Array, "{self.problem.n_params}"] | None): Initial parameters.
            return_best_params_history (bool): Whether to return history of best parameters.
            random_seed (int | None): Random seed for reproducibility.
            wall_times (list[int | float] | None): Wall time checkpoints.
            lb (Float[Array, "{self.problem.n_params}"] | None): Lower bounds for parameters.
            ub (Float[Array, "{self.problem.n_params}"] | None): Upper bounds for parameters.
            **bo_kwargs: Additional keyword arguments.

        Returns:
            tuple: Best parameters, history of best parameters, losses, wall time indices.
        """

        # Initialize variables
        best_params = jnp.zeros(self.problem.n_params)
        best_params_history = []
        best_loss = 1e10
        losses = []
        wall_time_indices = [] if wall_times is not None else None

        pbounds = (
            {f"x{i}": (lb[i], ub[i]) for i in range(self.problem.n_params)}
            if lb is not None and ub is not None
            else {f"x{i}": (-10, 10) for i in range(self.problem.n_params)}
        )
        
        @jax.jit
        def f(**kwargs):
            params = jnp.array([kwargs[f"x{i}"] for i in range(self.problem.n_params)])
            return -self.problem.objective_function(params)
        
        # Warmup f
        _ = f(**{f"x{i}": 0.0 for i in range(self.problem.n_params)})
        
        # Initialize wall_time_indices tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]


        # Initialize Bayesian Optimization
        # optimizer.maximize() internally does: suggest() -> evaluate -> register()
        optimizer = BO(
            f=f,
            pbounds=pbounds,
            random_state=random_seed,
            verbose=0,
            **bo_kwargs,
        )
        
        if init_params is not None:
            init_point = {f"x{i}": float(init_params[i]) for i in range(self.problem.n_params)}
            target = f(**init_point)
            optimizer.register(params=init_point, target=target)
        
        if wall_times is not None:
            start_time = time.time()
            i = 0
            
            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time

                # Record iteration index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(i)
                    wall_times_remaining.popleft()

                # Internal Bayesian Optimization step
                next_point = optimizer.suggest()
                target = f(**next_point)
                optimizer.register(params=next_point, target=target)

                loss = -target

                # Documentation
                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                # Update best params
                if loss < best_loss - 1e-4:
                    best_loss, best_params = loss, jnp.array(
                        [next_point[f"x{i}"] for i in range(self.problem.n_params)]
                    )
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(float(loss))
                i += 1
                
            # Fill remaining wall_times that weren't reached with final generation
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()
            
        else:
            for i in range(self.max_iterations):
                loss = -optimizer.maximize(init_points=0, n_iter=1).max["target"]
                
                # Documentation
                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                next_point = optimizer.max["params"]

                # Update best params
                if loss < best_loss - 1e-4:
                    best_loss, best_params = loss, jnp.array(
                        [next_point[f"x{i}"] for i in range(self.problem.n_params)]
                    )
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(float(loss))
        
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
                hyper_param_str="",  # TODO: add relevant hyperparameter(s) to arguments
                hyper_param_str_in_filename=False,
            )

        return best_params, best_params_history, losses, wall_time_indices
