from optimization.voyager.voyager_problem import VoyagerProblem
from optimization.algorithms.evox_pso import EvoxPSO
from optimization.config import create_parser

import jax

import cProfile
import pstats


def main():
    """Run Voyager optimization using EvoX PSO algorithm."""
    params = {
        "pop_size": 10,
        "n_generations": 10,
    }
    parser = create_parser(params, description="Voyager PSO Optimization")
    args = parser.parse_args()

    # Initialize problem
    vp = VoyagerProblem()

    # Initialize optimizer
    optimizer = EvoxPSO(problem=vp)

    # Run optimization
    optimizer.optimize(pop_size=args.pop_size, n_generations=args.n_generations)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.dump_stats("voyager_profile.prof")
    print("\nProfile saved to voyager_profile.prof")
    print("View with: snakeviz voyager_profile.prof")
