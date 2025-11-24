import os

os.environ["MPLCONFIGDIR"] = "/mnt/lustre/work/krenn/klz895/Differometor/tmp"

import jax
import jax.numpy as jnp
import torch
import time
from evox.algorithms import PSO
from evox.core import Problem as EvoxProblem
from evox.workflows import EvalMonitor, StdWorkflow
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from optimization import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
    j2t_numpy as j2t,
    t2j_numpy as t2j,
)


class EvoxPSO(OptimizationAlgorithm):
    """EvoX-based Particle Swarm Optimization algorithm.

    Implements PSO using the EvoX library with JAX backend. Handles batched
    evaluation of particles to manage memory efficiently.
    """

    algorithm_str: str = "evox_pso"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, problem: ContinuousProblem, batch_size: int = 5) -> None:
        """Initialize EvoX Particle Swarm Optimization.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            batch_size (int): Number of particles to evaluate simultaneously in each batch.
                Reduce this value if encountering out-of-memory errors. Defaults to 5.
        """
        self._problem = problem
        self._batch_size = batch_size

        # Define the problem in EvoX so it can be optimized
        class PSOProblem(EvoxProblem):
            def __init__(self, batch_size):
                super().__init__()
                self.batch_size = batch_size
                # vmap for a single batch
                self.vectorized_objective = jax.vmap(
                    problem.objective_function, in_axes=0
                )
                # warmup the function to compile it
                _ = self.vectorized_objective(
                    jnp.zeros((self.batch_size, problem.n_params))
                )

            def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
                # EvoX works in torch, this project in JAX
                jpop = t2j(pop)

                # Split population into batches to avoid OOM
                n_particles = jpop.shape[0]
                all_losses = []

                for i in range(0, n_particles, self.batch_size):
                    batch = jpop[i : i + self.batch_size]
                    batch_losses = self.vectorized_objective(batch)
                    all_losses.append(batch_losses)

                # Concatenate all batch results
                losses = jnp.concatenate(all_losses, axis=0)
                return j2t(losses)

        # ...and initiate it.
        self._pso_problem = PSOProblem(self._batch_size)

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        init_params_pop: Float[Array, "{pop_size} {self._problem.n_params}"]
        | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_time: int | float | None = None,
        pop_size: int = 100,
        n_generations: int | None = None,
        lb: Float[Array, "{self._problem.n_params}"] | None = None,
        ub: Float[Array, "{self._problem.n_params}"] | None = None,
        **pso_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_gens {self._problem.n_params}"],
        Float[Array, "n_gens"],
        Float[Array, "n_gens {pop_size}"],
    ]:
        """Run PSO optimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            init_params_pop (Float[Array, "pop_size n_params"] | None): Initial population of
                parameters. If None, randomly initialized within bounds. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                generation. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls both initial
                population generation and random coefficients during optimization. Defaults to None.
            wall_time (int | float | None): Maximum wall-clock time in seconds. If None, runs for
                n_generations. Defaults to None.
            pop_size (int): Number of particles in the swarm. Defaults to 100.
            n_generations (int | None): Number of generations to run. Required if wall_time
                is None. Defaults to None.
            lb (Float[Array, "n_params"] | None): Lower bounds for each parameter. If None,
                uses -10 for all parameters. Defaults to None.
            ub (Float[Array, "n_params"] | None): Upper bounds for each parameter. If None,
                uses 10 for all parameters. Defaults to None.
            **pso_kwargs: Additional keyword arguments passed to EvoX PSO constructor.
                Common options: w (float, inertia weight, default 0.6), phi_p (float,
                cognitive coefficient, default 2.5), phi_g (float, social coefficient,
                default 0.8).

        Returns:
            tuple: A 4-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_gens n_params"]): History of best
                  parameters per generation. Empty array if return_best_params_history=False.
                - losses (Float[Array, "n_gens"]): Best loss at each generation.
                - population_losses (Float[Array, "n_gens pop_size"]): Loss for each particle
                  at each generation.
        """
        # Set random seed if provided (affects both initialization and step randomness)
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Initiate monitor for loss tracking etc.
        monitor = EvalMonitor()

        # Convert bounds to torch tensors if needed
        if lb is None:
            lb = -10 * torch.ones(self._problem.n_params)
        elif isinstance(lb, jax.Array):
            lb = j2t(lb)
        if ub is None:
            ub = 10 * torch.ones(self._problem.n_params)
        elif isinstance(ub, jax.Array):
            ub = j2t(ub)

        # Initiate algorithm with hyper params
        algorithm = PSO(pop_size=pop_size, lb=lb, ub=ub, **pso_kwargs)

        # If initial population is provided, set it before init_step
        if init_params_pop is not None:
            # Convert to torch if needed
            if isinstance(init_params_pop, jax.Array):
                init_pop_torch = j2t(init_params_pop)
            else:
                init_pop_torch = init_params_pop
            # Only override the population - init_step will handle the rest
            algorithm.pop = init_pop_torch

        # This results in the workflow
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=self._pso_problem,
            monitor=monitor,
        )

        # Initialize: evaluates population and sets initial best values
        workflow.init_step()

        # Executing the algorithm itself
        best_params_history = []  # Shape: (n_steps, n_params)

        # Capture initial best params after init_step
        if return_best_params_history:
            best_params = t2j(monitor.topk_solutions)[0]
            best_params_history.append(best_params)

        # If there is no time limit:
        if wall_time is None:
            for _ in range(n_generations):
                workflow.step()
                if return_best_params_history:
                    best_params = t2j(monitor.topk_solutions)[0]
                    best_params_history.append(best_params)
        else:
            start_time = time.time()
            # With both time limit and generation limit:
            if n_generations is not None:
                for gen in range(n_generations):
                    if (time.time() - start_time) >= wall_time:
                        break
                    workflow.step()
                    if return_best_params_history:
                        best_params = t2j(monitor.topk_solutions)[0]
                        best_params_history.append(best_params)
            # With only time limit:
            else:
                while (time.time() - start_time) < wall_time:
                    workflow.step()
                    if return_best_params_history:
                        best_params = t2j(monitor.topk_solutions)[0]
                        best_params_history.append(best_params)

        # Extract results from monitor
        best_params = t2j(monitor.topk_solutions)[0]
        best_params_history = jnp.array(best_params_history)
        # jnp.array is used because fit_history is a list of tensors (doesnt have dlpack for t2j)
        population_losses = jnp.array(monitor.fit_history)
        losses = jnp.min(population_losses, axis=1)

        print("Best params history shape:")
        print(best_params_history.shape)

        hyper_param_str = f"_gen{n_generations}_pop{pop_size}"

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses,
                population_losses=population_losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=hyper_param_str,
            )

        return best_params, best_params_history, losses, population_losses
