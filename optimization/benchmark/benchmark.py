"""Benchmarking module for optimization algorithms.

This module provides tools to benchmark multiple optimization algorithms on a problem,
running each algorithm multiple times and computing various performance metrics.

Key classes:
    - Benchmark: Main benchmarking class that runs algorithms and evaluates performance
    - BenchmarkResult: Dataclass containing performance metrics across time steps
    - AlgorithmConfig: Configuration wrapper for algorithm + hyperparameters

Compatibility:
    - Works with any algorithm implementing OptimizationAlgorithm protocol
    - Requires algorithms to return: (best_params, best_params_history, losses, ...)
    - Requires return_best_params_history=True and wall_time parameters
"""

import numpy as np
import jax.numpy as jnp
import csv
from pathlib import Path
from datetime import datetime
from jaxtyping import Array, Float
from typing import Any, Dict, Optional
from dataclasses import dataclass, fields

from optimization import OptimizationAlgorithm, ContinuousProblem, AlgorithmType


@dataclass
class SingleMetric:
    """Metric with a single value per time step (not aggregated across runs)."""

    value: Float[Array, "n_time_steps"]


@dataclass
class AggregateMetric:
    """Metric aggregated across multiple runs with statistics."""

    mean: Float[Array, "n_time_steps"]
    std: Float[Array, "n_time_steps"]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run using one algorithm across multiple time steps.

    All metric arrays have shape (n_time_steps,) corresponding to self.wall_time_steps.

    Metrics are categorized by type:
    - SingleMetric: Properties of the distribution (single value, not aggregated)
    - AggregateMetric: Statistics computed across runs (mean, std, optionally median/min/max)
    """

    algorithm_name: str

    # Single-value metrics (distribution properties)
    fraction_of_success: SingleMetric
    min_loss: SingleMetric

    # Aggregate metrics (statistics across runs)
    avg_loss: AggregateMetric
    time_to_success: AggregateMetric
    calls_to_success: AggregateMetric
    solution_diversity_overall: AggregateMetric
    solution_diversity_nn: AggregateMetric

    time_per_call: AggregateMetric

    # Top-k metrics
    auc_top_1: SingleMetric
    auc_top_10: AggregateMetric


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
    """Benchmark multiple optimization algorithms on a problem.

    Runs multiple algorithms with different hyperparameter configurations,
    evaluates their performance across multiple runs, and computes various
    metrics including success rate, convergence time, and solution diversity.
    """

    def __init__(
        self,
        problem: ContinuousProblem,
        success_loss: float,
        configs: list[AlgorithmConfig],
        n_runs: int = 100,
        wall_time_steps: list[float] = [300],  # 5 minutes in seconds
    ):
        """Initialize the benchmark suite.

        Args:
            problem: The optimization problem to benchmark algorithms on
            success_loss: Loss threshold below which a run is considered successful
            configs: List of algorithm configurations to benchmark
            n_runs: Number of independent runs per algorithm configuration
            wall_time_steps: Time checkpoints (in seconds) at which to evaluate metrics
        """
        self._problem = problem
        self._success_loss = success_loss
        self._configs = configs
        self._n_runs = n_runs
        self._wall_time_steps = wall_time_steps
        # Warmup jit compilation
        _ = problem.objective_function(jnp.zeros(problem.n_params))

    # def _ensure_max_calls(self, config: AlgorithmConfig):
    #     match config.algorithm.algorithm_type:  # Maybe print a warning if overwritten?
    #         case AlgorithmType.GRADIENT_BASED:
    #             config.hyperparameters["max_iterations"] = self.max_calls # Each iteration is one call (i suppose)
    #         case AlgorithmType.EVOLUTIONARY:
    #             config.hyperparameters["n_generations"] = self.max_calls // config.hyperparameters.get("pop_size", 100)  # 100 is the default pop_size
    #     return config

    def _pad_to_max_length(self, arrays: list) -> jnp.ndarray:
        """Pad arrays to maximum length using final values.

        Handles arrays of any dimensionality by extending the first axis (iterations).
        Pads shorter sequences with their final value(s) to ensure all runs have
        the same number of iterations for vectorized processing.

        Args:
            arrays: List of JAX arrays with potentially different lengths along axis 0

        Returns:
            Stacked array with shape (n_runs, max_length, ...) where shorter
            runs are padded with their final values along the first axis
        """
        if not arrays:
            return jnp.array([])

        # Determine maximum length across all arrays
        max_len = max(len(arr) for arr in arrays)
        min_len = min(len(arr) for arr in arrays)

        # Warn if significant variation in iteration counts
        if max_len > 0 and (max_len - min_len) / min_len > 0.1:
            print(
                f"Warning: Iteration count varies by {(max_len - min_len) / min_len:.1%} ({min_len}-{max_len} iterations)"
            )

        padded = []
        for arr in arrays:
            if len(arr) < max_len:
                n_pad = max_len - len(arr)
                # Pad only the first axis (iterations): (0, n_pad)
                # Don't pad other axes: (0, 0) for each remaining dimension
                pad_width = ((0, n_pad),) + ((0, 0),) * (arr.ndim - 1)
                arr = jnp.pad(arr, pad_width, mode="edge")

            padded.append(arr)

        return jnp.array(padded)

    def _run_algorithm(self, config: AlgorithmConfig) -> BenchmarkResult:
        """Run a single algorithm configuration multiple times and evaluate results.

        Args:
            config: Algorithm configuration

        Returns:
            BenchmarkResult containing all metrics for this configuration
        """
        print(f"\n{'=' * 70}")
        print(f"Running: {config.name}")
        print(f"Hyperparameters: {config.hyperparameters}")
        print(f"Number of runs: {self._n_runs}")
        print(f"Wall time limit: {self._wall_time_steps[-1]:.1f}s")
        print(f"{'=' * 70}\n")

        all_best_params = []
        all_best_params_history = []
        all_losses = []

        for i_run in range(self._n_runs):
            print(f"Run {i_run + 1}/{self._n_runs}...", end=" ", flush=True)

            # Run optimization
            best_params, best_params_history, losses, *_ = config.algorithm.optimize(
                save_to_file=False,
                return_best_params_history=True,
                wall_time=self._wall_time_steps[-1],
                **config.hyperparameters,
            )

            # Store results
            all_best_params.append(best_params)
            all_best_params_history.append(best_params_history)
            all_losses.append(losses)

            # Print progress
            final_loss = losses[-1] if len(losses) > 0 else float("inf")
            print(f"Final loss: {final_loss:.6f}")

        print(f"\nCompleted {self._n_runs} runs for {config.name}")

        # Pad arrays to handle variable iteration counts due to wall time limits
        all_best_params = jnp.array(all_best_params)  # Should already be same shape
        all_best_params_history = self._pad_to_max_length(all_best_params_history)
        all_losses = self._pad_to_max_length(all_losses)

        # Evaluate results
        return self._evaluate_runs(
            config,
            all_best_params,
            all_best_params_history,
            all_losses,
        )

    def _evaluate_runs(
        self,
        config: AlgorithmConfig,
        all_best_params: Float[Array, "runs params"],
        all_best_params_history: Float[Array, "runs iterations params"],
        all_losses: Float[Array, "runs iterations"],
    ) -> BenchmarkResult:
        """Evaluate an algorithm's runs' results using vectorized operations.

        Args:
            config: Algorithm configuration
            all_best_params: Best parameters found during each run
            all_best_params_history: History of best parameters at each iteration for each run
            all_losses: Losses recorded during each run at each iteration

        Returns:
            BenchmarkResult containing all metrics as arrays
        """
        print("Evaluating runs...")
        print("Shapes:")
        print("all_best_params:", all_best_params.shape)
        print("all_best_params_history:", all_best_params_history.shape)
        print("all_losses:", all_losses.shape)

        n_runs, n_iterations = all_losses.shape
        n_time_steps = len(self._wall_time_steps)
        total_time = self._wall_time_steps[-1]
        time_per_iteration = total_time / n_iterations

        # Pre-allocate arrays for all metrics
        # Single-value metrics
        fraction_of_success_arr = jnp.zeros(n_time_steps)
        min_loss_arr = jnp.zeros(n_time_steps)
        auc_top_1_arr = jnp.zeros(n_time_steps)

        # Aggregate metrics (mean + std)
        avg_loss_mean = jnp.zeros(n_time_steps)
        avg_loss_std = jnp.zeros(n_time_steps)
        time_to_success_mean = jnp.zeros(n_time_steps)
        time_to_success_std = jnp.zeros(n_time_steps)
        calls_to_success_mean = jnp.zeros(n_time_steps)
        calls_to_success_std = jnp.zeros(n_time_steps)
        solution_diversity_overall_mean = jnp.zeros(n_time_steps)
        solution_diversity_overall_std = jnp.zeros(n_time_steps)
        solution_diversity_nn_mean = jnp.zeros(n_time_steps)
        solution_diversity_nn_std = jnp.zeros(n_time_steps)
        time_per_call_mean = jnp.zeros(n_time_steps)
        time_per_call_std = jnp.zeros(n_time_steps)
        auc_top_10_mean = jnp.zeros(n_time_steps)
        auc_top_10_std = jnp.zeros(n_time_steps)

        for i_time, wall_time in enumerate(self._wall_time_steps):
            # Get iteration index for this wall time
            max_iter = int(wall_time / time_per_iteration)
            max_iter = min(max_iter, n_iterations)

            # Slice losses up to this point
            losses_up_to_time = all_losses[:, :max_iter]  # Shape: (n_runs, max_iter)

            # FRACTION OF SUCCESS
            success_mask = (
                losses_up_to_time < self._success_loss
            )  # Shape: (n_runs, max_iter)
            has_success = jnp.any(success_mask, axis=1)  # Shape: (n_runs,)
            n_successes = jnp.sum(has_success)
            fraction_of_success_arr = fraction_of_success_arr.at[i_time].set(
                n_successes / n_runs
            )

            # TIME TO SUCCESS & CALLS TO SUCCESS (vectorized!)
            first_success_iter = jnp.argmax(success_mask, axis=1)  # Shape: (n_runs,)
            first_success_iter = jnp.where(has_success, first_success_iter, max_iter)

            # Calculate mean and std only for successful runs
            if n_successes > 0:
                successful_iters_only = first_success_iter[has_success]
                time_to_success_mean = time_to_success_mean.at[i_time].set(
                    jnp.mean(successful_iters_only) * time_per_iteration
                )
                time_to_success_std = time_to_success_std.at[i_time].set(
                    jnp.std(successful_iters_only) * time_per_iteration
                )
                calls_to_success_mean = calls_to_success_mean.at[i_time].set(
                    jnp.mean(successful_iters_only)
                )
                calls_to_success_std = calls_to_success_std.at[i_time].set(
                    jnp.std(successful_iters_only)
                )
            else:
                # No successful runs
                time_to_success_mean = time_to_success_mean.at[i_time].set(
                    float(max_iter * time_per_iteration)
                )
                time_to_success_std = time_to_success_std.at[i_time].set(0.0)
                calls_to_success_mean = calls_to_success_mean.at[i_time].set(
                    float(max_iter)
                )
                calls_to_success_std = calls_to_success_std.at[i_time].set(0.0)

            # MIN/AVG LOSS
            run_min_losses = jnp.min(losses_up_to_time, axis=1)  # Shape: (n_runs,)
            min_loss_arr = min_loss_arr.at[i_time].set(jnp.min(run_min_losses))
            avg_loss_mean = avg_loss_mean.at[i_time].set(jnp.mean(run_min_losses))
            avg_loss_std = avg_loss_std.at[i_time].set(jnp.std(run_min_losses))

            # TIME PER CALL (same for all runs at this time step, so std=0)
            time_per_call_mean = time_per_call_mean.at[i_time].set(
                time_per_iteration * 1000
            )  # in ms
            time_per_call_std = time_per_call_std.at[i_time].set(0.0)

            # AUC TOP 1: Area under curve for the single best run
            best_run_idx = jnp.argmin(run_min_losses)  # Find best run
            best_run_losses = losses_up_to_time[best_run_idx]  # Shape: (max_iter,)
            auc_top_1_arr = auc_top_1_arr.at[i_time].set(
                jnp.trapezoid(best_run_losses, dx=time_per_iteration)
            )

            # AUC TOP 10: Area under curve for top 10% runs (mean ± std)
            n_top = max(1, n_runs // 10)  # At least 1 run
            top_run_indices = jnp.argsort(run_min_losses)[
                :n_top
            ]  # Indices of best runs
            top_runs_losses = losses_up_to_time[
                top_run_indices
            ]  # Shape: (n_top, max_iter)

            # Compute AUC for each top run
            top_aucs = jnp.array(
                [
                    jnp.trapezoid(top_runs_losses[i], dx=time_per_iteration)
                    for i in range(n_top)
                ]
            )
            auc_top_10_mean = auc_top_10_mean.at[i_time].set(jnp.mean(top_aucs))
            auc_top_10_std = auc_top_10_std.at[i_time].set(jnp.std(top_aucs))

            # SOLUTION DIVERSITY
            # Get best parameters at max_iter for all successful runs
            params_at_max_iter = all_best_params_history[
                :, max_iter - 1, :
            ]  # Shape: (n_runs, n_params)
            successful_params = params_at_max_iter[
                has_success
            ]  # Shape: (n_successful, n_params)

            # Calculate diversity only if we have at least 2 successful solutions
            if n_successes > 1:
                diversity_overall_vals, diversity_nn_vals = (
                    self._calculate_diversity_with_std(successful_params, n_successes)
                )
                solution_diversity_overall_mean = solution_diversity_overall_mean.at[
                    i_time
                ].set(diversity_overall_vals[0])
                solution_diversity_overall_std = solution_diversity_overall_std.at[
                    i_time
                ].set(diversity_overall_vals[1])
                solution_diversity_nn_mean = solution_diversity_nn_mean.at[i_time].set(
                    diversity_nn_vals[0]
                )
                solution_diversity_nn_std = solution_diversity_nn_std.at[i_time].set(
                    diversity_nn_vals[1]
                )
            else:
                solution_diversity_overall_mean = solution_diversity_overall_mean.at[
                    i_time
                ].set(0.0)
                solution_diversity_overall_std = solution_diversity_overall_std.at[
                    i_time
                ].set(0.0)
                solution_diversity_nn_mean = solution_diversity_nn_mean.at[i_time].set(
                    0.0
                )
                solution_diversity_nn_std = solution_diversity_nn_std.at[i_time].set(
                    0.0
                )

        return BenchmarkResult(
            algorithm_name=config.name,
            fraction_of_success=SingleMetric(value=fraction_of_success_arr),
            min_loss=SingleMetric(value=min_loss_arr),
            avg_loss=AggregateMetric(mean=avg_loss_mean, std=avg_loss_std),
            time_to_success=AggregateMetric(
                mean=time_to_success_mean, std=time_to_success_std
            ),
            calls_to_success=AggregateMetric(
                mean=calls_to_success_mean, std=calls_to_success_std
            ),
            solution_diversity_overall=AggregateMetric(
                mean=solution_diversity_overall_mean, std=solution_diversity_overall_std
            ),
            solution_diversity_nn=AggregateMetric(
                mean=solution_diversity_nn_mean, std=solution_diversity_nn_std
            ),
            time_per_call=AggregateMetric(
                mean=time_per_call_mean, std=time_per_call_std
            ),
            auc_top_1=SingleMetric(value=auc_top_1_arr),
            auc_top_10=AggregateMetric(mean=auc_top_10_mean, std=auc_top_10_std),
        )

    def _calculate_diversity_with_std(
        self, successful_params: Float[Array, "n_successful n_params"], n_successes: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Calculate solution diversity metrics with standard deviations.

        Args:
            successful_params: Parameters of successful solutions
            n_successes: Number of successful solutions

        Returns:
            Tuple of ((overall_diversity_mean, overall_diversity_std),
                     (nearest_neighbor_diversity_mean, nearest_neighbor_diversity_std))
        """
        # Calculate pairwise distances between all successful solutions
        diff = successful_params[:, None, :] - successful_params[None, :, :]
        distances = jnp.linalg.norm(diff, axis=2)  # Shape: (n_succ, n_succ)

        # OVERALL DIVERSITY: Mean and std of pairwise distances (excluding diagonal)
        mask = ~jnp.eye(n_successes, dtype=bool)
        pairwise_distances = distances[mask]
        diversity_overall_mean = jnp.mean(pairwise_distances)
        diversity_overall_std = jnp.std(pairwise_distances)

        # NEAREST NEIGHBOR DIVERSITY: Mean and std of distances to nearest neighbors
        distances_no_diag = jnp.where(
            jnp.eye(n_successes, dtype=bool), jnp.inf, distances
        )
        nearest_neighbor_distances = jnp.min(distances_no_diag, axis=1)
        diversity_nn_mean = jnp.mean(nearest_neighbor_distances)
        diversity_nn_std = jnp.std(nearest_neighbor_distances)

        return (diversity_overall_mean, diversity_overall_std), (
            diversity_nn_mean,
            diversity_nn_std,
        )

    def run_benchmark(self, save_to_file: bool = True) -> list[BenchmarkResult]:
        """Run benchmark for all algorithm configurations and return results.

        Executes each algorithm configuration for the specified number of runs,
        evaluates performance metrics at each time checkpoint, and optionally
        saves results to a timestamped CSV file.

        Args:
            save_to_file: Whether to save results to CSV file (default: True)

        Returns:
            List of BenchmarkResult, one for each algorithm configuration
        """
        print("\n" + "=" * 70)
        print("BENCHMARK SUITE")
        print("=" * 70)
        print(
            f"Problem: {self._problem._name if hasattr(self._problem, '_name') else 'Unknown'}"
        )
        print(f"Success threshold: {self._success_loss}")
        print(f"Number of algorithms: {len(self._configs)}")
        print(f"Runs per algorithm: {self._n_runs}")
        print(f"Wall time checkpoints: {self._wall_time_steps}")
        print("=" * 70)

        results = []

        for i, config in enumerate(self._configs, 1):
            print(f"\n[{i}/{len(self._configs)}] Benchmarking: {config.name}")
            result = self._run_algorithm(config)
            results.append(result)

            # Print summary for this algorithm
            print(f"\n--- Summary for {config.name} ---")
            print(
                f"  Success rate (final): {float(result.fraction_of_success.value[-1]):.1%}"
            )
            print(f"  Min loss (final): {float(result.min_loss.value[-1]):.6f}")
            print(
                f"  Avg loss (final): {float(result.avg_loss.mean[-1]):.6f} ± {float(result.avg_loss.std[-1]):.6f}"
            )
            print(
                f"  Time to success: {float(result.time_to_success.mean[-1]):.2f} ± {float(result.time_to_success.std[-1]):.2f}s"
            )
            print(
                f"  Diversity (overall): {float(result.solution_diversity_overall.mean[-1]):.4f} ± {float(result.solution_diversity_overall.std[-1]):.4f}"
            )

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

        # Save results to file if requested
        if save_to_file:
            self._save_results_to_csv(results)

        return results

    def print_summary(self, results: list[BenchmarkResult]):
        """Print a summary comparison of all algorithm results.

        Displays a formatted table comparing algorithms on key metrics
        including success rate, min/avg loss, time to success, and diversity.

        Args:
            results: List of BenchmarkResult from run_benchmark()
        """
        print("\n" + "=" * 90)
        print("BENCHMARK SUMMARY (at final time step)")
        print("=" * 90)

        # Header
        print(
            f"{'Algorithm':<20} {'Success%':>10} {'Min Loss':>12} {'Avg Loss':>15} {'Time(s)':>15} {'Diversity':>15}"
        )
        print("-" * 90)

        # Results for each algorithm
        for result in results:
            name = result.algorithm_name[:19]  # Truncate if too long
            success = float(result.fraction_of_success.value[-1]) * 100
            min_loss = float(result.min_loss.value[-1])
            avg_loss_str = f"{float(result.avg_loss.mean[-1]):.4f}±{float(result.avg_loss.std[-1]):.4f}"
            time_str = f"{float(result.time_to_success.mean[-1]):.2f}±{float(result.time_to_success.std[-1]):.2f}"
            diversity_str = f"{float(result.solution_diversity_overall.mean[-1]):.3f}±{float(result.solution_diversity_overall.std[-1]):.3f}"

            print(
                f"{name:<20} {success:>9.1f}% {min_loss:>12.6f} {avg_loss_str:>15} {time_str:>15} {diversity_str:>15}"
            )

        print("=" * 90)

    def _save_results_to_csv(self, results: list[BenchmarkResult]):
        """Save benchmark results to a CSV file with timestamp.

        Creates a ./benchmarks/ directory and saves results with filename
        format: benchmark_{problem_name}_{timestamp}.csv

        Args:
            results: List of BenchmarkResult from run_benchmark()
        """
        # Create benchmarks directory
        output_dir = Path("./benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = (
            self._problem._name if hasattr(self._problem, "_name") else "problem"
        )
        output_path = output_dir / f"benchmark_{problem_name}_{timestamp}.csv"

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # 1. UNIFORM HEADER GENERATION
            # Every metric gets a _mean and _std suffix, even single ones,
            # making the CSV easier to parse
            header = ["algorithm_name", "wall_time_step"]
            for field in fields(BenchmarkResult):
                if field.name == "algorithm_name":
                    continue
                # Regardless of type, we reserve two columns
                header.append(f"{field.name}_mean")
                header.append(f"{field.name}_std")

            writer.writerow(header)

            # 2. UNIFORM DATA WRITING
            for result in results:
                n_time_steps = len(self._wall_time_steps)

                for i in range(n_time_steps):
                    row = [result.algorithm_name, float(self._wall_time_steps[i])]

                    for field in fields(BenchmarkResult):
                        if field.name == "algorithm_name":
                            continue
                        metric = getattr(result, field.name)

                        if isinstance(metric, SingleMetric):
                            # For SingleMetric, the "Std" is always 0.0
                            row.append(float(metric.value[i]))
                            row.append(0.0)
                        elif isinstance(metric, AggregateMetric):
                            row.append(float(metric.mean[i]))
                            row.append(float(metric.std[i]))

                    writer.writerow(row)

        print(f"\nBenchmark results saved to: {output_path}")
