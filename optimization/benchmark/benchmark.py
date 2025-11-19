import numpy as np
import jax.numpy as jnp
from typing import Any, Dict, Optional
from dataclasses import dataclass

from optimization import OptimizationAlgorithm, ContinuousProblem, AlgorithmType


@dataclass
class BenchmarkResult:
    """Results from a benchmark run using one algorithm each timestep.
    """

    algorithm_name: str
    fraction_of_success: list[float]
    time_to_success: list[float]
    calls_to_success: list[int]
    time_per_call: list[float]
    min_loss: list[float]
    avg_loss: list[float]
    auc_top_1: list[float]
    auc_top_10: list[float]
    solution_diversity_overall: list[float]
    solution_diversity_nn: list[float]
    


class AlgorithmConfig:
    """Configuration for an algorithm to benchmark."""

    def __init__(
        self,
        algorithm: OptimizationAlgorithm,
        hyperparameters: Dict[str, Any],
        name: Optional[str] = None,
    ):
        """
        Args:
            algorithm: The algorithm instance to benchmark
            hyperparameters: Dictionary of hyperparameters to pass to optimize()
            name: Optional custom name for this configuration
        """
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.name = name or algorithm.algorithm_str


class Benchmark:
    def __init__(
        self,
        problem: ContinuousProblem,
        configs: list[AlgorithmConfig],
        n_runs: int = 100,
        wall_time: float = 10_800, # 3 hours in seconds
    ):
        self._problem = problem
        self.n_runs = n_runs
        self.wall_time = wall_time

        # Warmup jit compilation
        _ = problem.objective_function(jnp.zeros(problem.n_params))
        
        
    # def _ensure_max_calls(self, config: AlgorithmConfig):
    #     match config.algorithm.algorithm_type:  # Maybe print a warning if overwritten?
    #         case AlgorithmType.GRADIENT_BASED:
    #             config.hyperparameters["max_iterations"] = self.max_calls # Each iteration is one call (i suppose)
    #         case AlgorithmType.EVOLUTIONARY:
    #             config.hyperparameters["n_generations"] = self.max_calls // config.hyperparameters.get("pop_size", 100)  # 100 is the default pop_size
    #     return config

    def _run_single(self, config: AlgorithmConfig, run_id: int) -> BenchmarkResult:
        """Run a single optimization run.

        Args:
            config: Algorithm configuration
            run_id: Run identifier
            save_to_file: Whether to save results to file

        Returns:
            BenchmarkResult for this run
        """
        result = BenchmarkResult
        
        for run_id in range(self.n_runs):
            best_params, losses, *_ = config.algorithm.optimize(save_to_file=False, **config.hyperparameters)

    def run_all(self):
        for algorithm in self._algorithms:
            pass  # loop TODO
