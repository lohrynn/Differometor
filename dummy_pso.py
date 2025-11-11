from optimization.voyager.voyager_problem import VoyagerProblem
from optimization.protocols import ContinuousProblem
from optimization.algorithms.evox_pso import EvoxPSO
from optimization.config import create_parser
from jaxtyping import Float, Array

import jax.numpy as jnp
from jax import random
import jax
import time

def main():
    """Run Voyager optimization using EvoX PSO algorithm."""
    params = {
        "pop_size": 10,
        "n_generations": 10,
    }
    parser = create_parser(params, description="Voyager PSO Optimization")
    args = parser.parse_args()

    class DummyProblem(ContinuousProblem):
        

        def __init__(self):
            super().__init__()
            n_params = 48
            self._n_params = n_params
            A = random.normal(random.key(0), (n_params,n_params))
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

    # Initialize problem
    dp = DummyProblem()

    # Initialize optimizer
    optimizer = EvoxPSO(problem=dp)

    # Run optimization
    # with jax.profiler.trace("./tmp/jax-trace", create_perfetto_link=True):

    start = time.time()
    optimizer.optimize(pop_size=args.pop_size, n_generations=args.n_generations)
    end = time.time()
    print(f"Optimization took {end - start} seconds.")


if __name__ == "__main__":
    main()
