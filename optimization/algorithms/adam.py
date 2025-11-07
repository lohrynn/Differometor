import jax.numpy as jnp
import jax
import numpy as np
import optax
from jaxtyping import Array, Float

from optimization.protocols import ContinuousProblem, OptimizationAlgorithm


class AdamGD(OptimizationAlgorithm):
    algorithm_str: str = "adam"

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
        self._best_params: Float[Array, "{self._problem.n_params}"] = jnp.array(
            np.random.uniform(-10, 10, self._problem.n_params)
        )
        self._losses = []
        self._best_loss = 1e10

        self._grad_fn = jax.jit(jax.value_and_grad(self._problem.objective_function))

    def optimize(
        self, save_to_file: bool = True, learning_rate: float = 0.1, **adam_kwargs
    ):
        """Run optimization with Adam.

        Args:
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.1
            **adam_kwargs: Additional keyword arguments passed to optax.adam()
                          (Default: b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, nesterov=False)
        """
        # warmup the function to compile it
        _ = self._grad_fn(self._best_params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(self._best_params)

        params, no_improve_count, losses = self._best_params, 0, []

        for i in range(self.max_iterations):
            loss, grads = self._grad_fn(params)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss}")

            if loss < self._best_loss - 1e-4:
                self._best_loss, self._best_params, no_improve_count = loss, params, 0
                print(f"Iteration {i}: New best loss = {loss}")
            else:
                no_improve_count += 1

            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

            # if the loss has not improved (< best_loss - 1e-4) over 1000 iterations, stop the optimization
            if no_improve_count > self.patience:
                break

        self._losses = jnp.array(losses)

        if save_to_file:
            self._problem.output_to_files(
                best_params=self._best_params,
                losses=self._losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"lr{learning_rate}",
            )  # TODO conditionally add more hyperparameters to string

        return self._best_params, losses
