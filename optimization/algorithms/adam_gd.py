import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from optimization.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)


class AdamGD(OptimizationAlgorithm):
    """Adam Gradient Descent optimization algorithm.

    Implements gradient-based optimization using the Adam optimizer from Optax.
    Includes gradient clipping and early stopping based on patience.
    """

    algorithm_str: str = "adam"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
        max_iterations: int = 50000,
        patience: int = 1000,
    ) -> None:
        """Initialize Adam Gradient Descent optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            max_iterations (int): Maximum number of optimization iterations. Defaults to 50,000.
            patience (int): Stop if no improvement (>1e-4) after this many iterations.
                Only applies when wall_time is not set. Defaults to 1,000.
        """
        self._problem = problem
        self.max_iterations = max_iterations
        self.patience = patience

        self._grad_fn = jax.jit(jax.value_and_grad(self._problem.objective_function))

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_time: int | float | None = None,
        learning_rate: float = 0.1,
        **adam_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
    ]:
        """Run Adam gradient descent optimization.ptimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            init_params (Float[Array, "n_params"] | None): Initial parameters. If None,
                randomly initialized in range [-10, 10]. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                iteration. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls initial
                parameter generation when init_params is None. Defaults to None.
            wall_time (int | float | None): Maximum wall-clock time in seconds. If None, runs for
                max_iterations or until patience is exceeded. Defaults to None.
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.1.
            **adam_kwargs: Additional keyword arguments passed to optax.adam().
                Common options: b1 (float, default 0.9), b2 (float, default 0.999),
                eps (float, default 1e-8), eps_root (float, default 0.0),
                nesterov (bool, default False).

        Returns:
            tuple: A 3-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_iters n_params"] | None): History of
                  best parameters per iteration. None if return_best_params_history=False.
                - losses (Float[Array, "n_iters"]): Loss at each iteration.
        """
        # Set random seed if provided (affects initial parameter generation)
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize parameters
        best_params: Float[Array, "{self._problem.n_params}"] = (
            jnp.array(np.random.uniform(-10, 10, self._problem.n_params))
            if init_params is None
            else init_params
        )

        # warmup the function to compile it
        _ = self._grad_fn(best_params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(best_params)

        params, losses = best_params, []
        best_params_history = []  # later shape: (n_iterations, n_params)
        best_loss = 1e10

        # Separate loops for wall-time constrained vs iteration/patience constrained
        if wall_time is not None:
            # Wall-time constrained: ignore max_iterations and patience
            start_time = time.time()
            i = 0
            while (time.time() - start_time) < wall_time:
                loss, grads = self._grad_fn(params)

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss, best_params = loss, params
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                updates, optimizer_state = optimizer.update(
                    grads, optimizer_state, params
                )
                params = optax.apply_updates(params, updates)
                losses.append(float(loss))
                i += 1
        else:
            # Iteration/patience constrained
            no_improve_count = 0
            for i in range(self.max_iterations):
                loss, grads = self._grad_fn(params)

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss, best_params, no_improve_count = loss, params, 0
                    print(f"Iteration {i}: New best loss = {loss}")
                else:
                    no_improve_count += 1

                if return_best_params_history:
                    best_params_history.append(best_params)

                updates, optimizer_state = optimizer.update(
                    grads, optimizer_state, params
                )
                params = optax.apply_updates(params, updates)
                losses.append(float(loss))

                if no_improve_count > self.patience:
                    break

        losses = jnp.array(losses)
        best_params_history = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"lr{learning_rate}",
            )  # TODO maybe conditionally add more hyperparameters to string

        return best_params, best_params_history, losses
