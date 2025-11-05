import os

os.environ["MPLCONFIGDIR"] = "/mnt/lustre/work/krenn/klz895/Differometor/tmp"
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
import torch
from evox.algorithms import PSO
from evox.workflows import StdWorkflow, EvalMonitor
from evox.core import Problem as EvoxProblem

from optimization.optimization_protocols import (
    t2j,
    j2t,
    ContinuousProblem,
    OptimizationAlgorithm,
)


class EvoxPSO(OptimizationAlgorithm):
    algorithm_str: str = "evox_pso"

    def __init__(self, problem: ContinuousProblem):
        """Initialize EvoX Particle Swarm Optimization (PSO)

        Args:
            problem (ContinuousProblem): The problem being optimized
        """
        self._problem = problem

        # Define the problem in EvoX so it can be optimized
        class PSOProblem(EvoxProblem):
            def __init__(self):
                super().__init__()
                # vmap for whole population
                self.vectorized_objective = jax.vmap(
                    problem.objective_function, in_axes=0
                )

            def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
                # EvoX works in torch, this project in JAX
                return j2t(self.vectorized_objective(t2j(pop)))

        # ...and initiate it.
        self._pso_problem = PSOProblem()

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
            n_generations (int): Number of generations to run. Defaults to 1000
            **pso_kwargs: Additional keyword arguments passed to PSO()
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
        for _ in range(n_generations):
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
