import os

os.environ["MPLCONFIGDIR"] = "/mnt/lustre/work/krenn/klz895/Differometor/tmp"

import jax
import jax.numpy as jnp
import torch
from evox.algorithms import PSO
from evox.core import Problem as EvoxProblem
from evox.workflows import EvalMonitor, StdWorkflow
from jaxtyping import Array, Float

from optimization.protocols import ContinuousProblem, OptimizationAlgorithm
from optimization.utils import j2t_numpy as j2t
from optimization.utils import t2j_numpy as t2j


class EvoxPSO(OptimizationAlgorithm):
    algorithm_str: str = "evox_pso"

    def __init__(self, problem: ContinuousProblem, batch_size: int = 5):
        """Initialize EvoX Particle Swarm Optimization (PSO)

        Args:
            problem (ContinuousProblem): The problem being optimized
            batch_size (int): Number of particles to evaluate simultaneously.
                             Reduce this if you get OOM errors. Defaults to 5.
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

    def optimize(
        self,
        save_to_file: bool = True,
        pop_size: int = 100,
        n_generations: int = 100,
        lb: Float[Array, "{self._problem.n_params}"] = None,
        ub: Float[Array, "{self._problem.n_params}"] = None,
        **pso_kwargs,
    ):
        """Run optimization with EvoX PSO.

        Args:
            pop_size (int): Number of particles in the swarm. Defaults to 100
            n_generations (int): Number of generations to run. Defaults to 100
            **pso_kwargs: Additional keyword arguments passed to PSO(). (w=0.6, phi_p=2.5, phi_g=0.8)
        """
        # Initiate monitor for loss tracking etc.
        monitor = EvalMonitor()
        # Ensure lb and ub are of the type (Array type accepts both)
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

        # This results in the workflow
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=self._pso_problem,
            monitor=monitor,
        )

        # Executing the algorithm itself
        for gen in range(n_generations):
            workflow.step()

        # Extract results from monitor
        best_params = t2j(monitor.topk_solutions)[0]
        # jnp.array is used because fit_history is a list of tensors (doesnt have dlpack for t2j)
        population_losses = jnp.array(monitor.fit_history)
        losses = jnp.min(population_losses, axis=1)

        hyper_param_str = f"_gen{n_generations}_pop{pop_size}"

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses,
                population_losses=population_losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=hyper_param_str,
            )

        return best_params, losses, population_losses
