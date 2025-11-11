from optimization.voyager.voyager_problem import VoyagerProblem
from optimization.protocols import ContinuousProblem
from optimization.algorithms.evox_pso import EvoxPSO
from jaxtyping import Float, Array

import jax.numpy as jnp
from jax import random
import jax
import time
import numpy as np
import matplotlib.pyplot as plt


class DummyProblem(ContinuousProblem):
    
    def __init__(self):
        super().__init__()
        n_params = 48
        self._n_params = n_params
        A = random.normal(random.key(0), (n_params, n_params))
        b = random.normal(random.key(1), (n_params,))

        @jax.jit
        def objective_function(optimized_parameters: Array) -> Float:
            return 0.5 * optimized_parameters.T @ (A @ A.T + jnp.eye(48)) @ optimized_parameters + b.T @ optimized_parameters

        self.objective_function = objective_function
        
    @property
    def optimization_pairs(self):
        return [("x"+str(i), "x"+str(i)) for i in range(48)]

    @property
    def n_params(self):
        return self._n_params
    
    def output_to_files(self, *args, **kwargs):
        pass


def benchmark_pop_size():
    """Benchmark varying population sizes with fixed generations."""
    print("Benchmarking population size scaling...")
    # Use linspace for equally spaced samples - focus on 50-400 range with 150 samples
    pop_sizes = np.linspace(50, 400, 150).astype(int)
    pop_sizes = np.unique(pop_sizes)  # Remove duplicates from rounding
    n_generations = 20
    times = []
    
    for pop_size in pop_sizes:
        print(f"Running pop_size={pop_size}...")
        dp = DummyProblem()
        optimizer = EvoxPSO(problem=dp)
        
        # Warmup
        optimizer.optimize(pop_size=int(pop_size), n_generations=2)
        
        # Measure
        start = time.time()
        optimizer.optimize(pop_size=int(pop_size), n_generations=n_generations)
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"Pop size {pop_size}: {elapsed:.3f} seconds")
    
    return np.array(pop_sizes), np.array(times)


def benchmark_generations():
    """Benchmark varying generation counts with fixed population size."""
    print("\nBenchmarking generation count scaling...")
    # Use linspace for equally spaced samples - focus on 50-400 range with 150 samples
    n_generations_list = np.linspace(50, 400, 150).astype(int)
    n_generations_list = np.unique(n_generations_list)  # Remove duplicates
    pop_size = 20
    times = []
    
    for n_gen in n_generations_list:
        print(f"Running n_generations={n_gen}...")
        dp = DummyProblem()
        optimizer = EvoxPSO(problem=dp)
        
        # Warmup
        optimizer.optimize(pop_size=pop_size, n_generations=2)
        
        # Measure
        start = time.time()
        optimizer.optimize(pop_size=pop_size, n_generations=int(n_gen))
        end = time.time()
        elapsed = end - start
        times.append(elapsed)
        print(f"Generations {n_gen}: {elapsed:.3f} seconds")
    
    return np.array(n_generations_list), np.array(times)


def plot_scaling(x_vals, times, xlabel, title, filename):
    """Plot scaling without fit line."""
    # Create plot with linear scales and line-only style for dense data
    plt.figure(figsize=(7, 5))
    plt.plot(x_vals, times, '-', label="Measured time", color="C0", linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel("Execution time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved plot to {filename}")
    plt.show()


def main():
    """Run benchmarks and create plots."""
    # Benchmark population size scaling
    pop_sizes, pop_times = benchmark_pop_size()
    plot_scaling(
        pop_sizes, 
        pop_times,
        xlabel="Population Size",
        title="PSO Execution Time vs Population Size (50-400, 5 generations)",
        filename="pso_pop_size_scaling_50_400.png"
    )
    
    # Benchmark generation count scaling
    generations, gen_times = benchmark_generations()
    plot_scaling(
        generations,
        gen_times,
        xlabel="Number of Generations",
        title="PSO Execution Time vs Generations (50-400, pop_size=20)",
        filename="pso_generation_scaling_50_400.png"
    )


if __name__ == "__main__":
    main()
