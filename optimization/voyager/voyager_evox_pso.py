import cProfile
import pstats

from optimization import EvoxPSO, VoyagerProblem, create_parser


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

    # Run optimization
    optimizer.optimize(pop_size=args.pop_size, n_generations=args.n_generations)


if __name__ == "__main__":
    main()
