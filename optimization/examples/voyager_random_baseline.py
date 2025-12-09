from optimization.problems.voyager_problem import VoyagerProblem

problem = VoyagerProblem(use_sigmoid_bounding=False)


# Prints statistics and returns dict with mean, std, min, max, median

means = []

for i in range(40,60):
    means.append(problem.estimate_random_baseline(n_samples=1000, batch_size=125, seed=i,)['mean'])

mean_baseline = sum(means) / len(means)
std_baseline = (sum((x - mean_baseline) ** 2 for x in means) / len(means)) ** 0.5

print(f"Random baseline over 20 runs (1000 samples each):")
print(f"Mean: {mean_baseline:.6f}")
print(f"Std: {std_baseline:.6f}")
print(f"Min: {min(means):.6f}")
print(f"Max: {max(means):.6f}")
print(f"Median: {sorted(means)[len(means)//2]:.6f}")