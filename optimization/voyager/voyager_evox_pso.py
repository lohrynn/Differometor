from optimization.voyager.voyager_problem import VoyagerProblem
from optimization.algorithms.evox_pso import EvoxPSO


def main():
    """Run Voyager optimization using EvoX PSO algorithm."""
    # Initialize problem
    vp = VoyagerProblem()

    # Initialize optimizer
    optimizer = EvoxPSO(problem=vp)

    pop_size = 20
    num_generations = 2

    # Run optimization
    optimizer.optimize(pop_size=pop_size, n_generations=num_generations)


if __name__ == "__main__":
    main()
