from optimization import EvoxPSO, AdamGD, VoyagerProblem, Benchmark, AlgorithmConfig

# Setup
problem = VoyagerProblem()
configs = [
    AlgorithmConfig(EvoxPSO(problem, batch_size=125), {"pop_size": 500}, "PSO-500"),
    AlgorithmConfig(AdamGD(problem), {"learning_rate": 0.1}, "Adam-0.1"),
]

# Run benchmark
benchmark = Benchmark(
    problem=problem,
    success_loss=0,
    configs=configs,
    n_runs=15,
    wall_time_steps=[120, 300, 600, 1200],
)

results = benchmark.run_benchmark(save_csv=True, save_run_data=True)
benchmark.print_summary(results)
