import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from jaxtyping import Array, Float

from optimization.protocols import ContinuousProblem, OptimizationAlgorithm, AlgorithmType


class AdamGD(OptimizationAlgorithm):
    algorithm_str: str = "adam"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
        max_iterations: int = 50000,
        patience: int = 1000,
    ):
        """Initialize gradient descent

        Args:
            problem (ContinuousProblem): The problem being optimized
            max_iterations (int): Maximum number of iterations. Defaults to 50,000
            patience (int): Stop if no improvement after this many iterations. Defaults to
        """
        self._problem = problem
        self.max_iterations = max_iterations
        self.patience = patience
        best_params: Float[Array, "{self._problem.n_params}"] = jnp.array(
            np.random.uniform(-10, 10, self._problem.n_params)
        )
        losses = []
        best_loss = 1e10

        self._grad_fn = jax.jit(jax.value_and_grad(self._problem.objective_function))

    def optimize(
        self,
        save_to_file: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] = None,
        return_best_params_history: bool = False,
        wall_time: float = None,
        learning_rate: float = 0.1,
        **adam_kwargs
    ):
        """Run optimization with Adam.

        Args:
            save_to_file (bool): Whether to save results to file. Defaults to True
            return_best_params_history (bool): Whether to track best params at each iteration. Defaults to False
            wall_time (float): Maximum wall time in seconds. Defaults to None
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.1
            **adam_kwargs: Additional keyword arguments passed to optax.adam()
                          (Default: b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, nesterov=False)
        """
        
        # Initialize parameters
        best_params: Float[Array, "{self._problem.n_params}"] = jnp.array(
            np.random.uniform(-10, 10, self._problem.n_params)
        ) if init_params is None else init_params
        
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

                updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
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

                updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
                params = optax.apply_updates(params, updates)
                losses.append(float(loss))

                if no_improve_count > self.patience:
                    break

        losses = jnp.array(losses)
        best_params_history = jnp.array(best_params_history) if return_best_params_history else None

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"lr{learning_rate}",
            )  # TODO maybe conditionally add more hyperparameters to string

        return best_params, best_params_history, losses
