import cProfile
import pstats

from optimization import (
    EvoxPSO,
    VoyagerProblem,
    create_parser,
)


def main():
    """Run Voyager optimization using EvoX PSO algorithm."""
    params = {
        "batch_size": 5,
        "pop_size": 50,
        "n_generations": 10,
    }
    parser = create_parser(params, description="Voyager PSO Optimization")
    args = parser.parse_args()

    # Initialize problem
    vp = VoyagerProblem()

    # Initialize optimizer with batch_size to control memory usage
    # Start with batch_size=5, increase if you have memory to spare
    optimizer = EvoxPSO(problem=vp, batch_size=args.batch_size)

    optimizer.optimize(
        pop_size=args.pop_size,
        n_generations=args.n_generations,
    )

    # param_space = {
    #     "pop_size": [20, 30, 50],
    #     "n_generations": [10, 20],
    # }

    # visualizer = HyperparameterVisualizer(algorithm=optimizer, param_space=param_space, problem=vp)

    # print("\nGenerating loss curves grid...")
    # visualizer.visualize_loss_grids(figsize_per_subplot=(3, 2.5))

    # print("\nGenerating sensitivity curves grid...")
    # visualizer.visualize_sensitivity_grids(figsize_per_subplot=(3.5, 2.5))


if __name__ == "__main__":
    main()
