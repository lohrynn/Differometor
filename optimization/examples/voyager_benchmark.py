from optimization import EvoxPSO, AdamGD, VoyagerProblem, Benchmark, AlgorithmConfig

# Setup
problem = VoyagerProblem()
configs = [
    AlgorithmConfig(EvoxPSO(problem, batch_size=40), {"pop_size": 80}, "PSO-80"),
    AlgorithmConfig(AdamGD(problem), {"learning_rate": 0.1}, "Adam-0.1"),
]

# Run benchmark
benchmark = Benchmark(
    problem=problem,
    success_loss=100,
    configs=configs,
    n_runs=2,
    wall_time_steps=[10, 30, 60],
)

results = benchmark.run_benchmark(save_to_file=True)
benchmark.print_summary(results)
