import differometor as df
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
import matplotlib.pyplot as plt
import numpy as np
import jax
import optax
import pygad
import json
import os


### Calculate the target sensitivity ###
#--------------------------------------#

# use a predefined Voyager setup with one noise detector and two signal detectors
S, component_property_pairs = voyager()

# set the frequency range
frequencies = np.logspace(np.log10(20), np.log10(5000), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
powers = powers[0] - powers[1]

# calculate the sensitivity
target_sensitivity = noise / np.abs(powers)


### Start from random parameters and optimize the sensitivity ###
#---------------------------------------------------------------#


# specify the ranges for the properties to be optimized
property_bounds = {
    "reflectivity": [0, 1],
    "tuning": [0, 90],
    "db": [0.01, 20],
    "angle": [-180, 180],
    "power": [0.01, 200],
    "mass": [0.01, 200],
    "length": [1, 4000],
    "phase": [-180, 180],
}

# select properties to be optimized
optimized_properties = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length", "phase"]
optimization_pairs = []
for pair in component_property_pairs:
    if pair[1] in optimized_properties:
        optimization_pairs.append(pair)

# build the setup once and then reuse it during the optimization
simulation_arrays, detector_ports, *_ = df.run_build_step(
    S,
    [("f", "frequency")],
    frequencies,
    optimization_pairs,
)

# calculate the bounds for the properties to be optimized
bounds = np.array([[
    property_bounds[pair[1]][0], 
    property_bounds[pair[1]][1]] for pair in optimization_pairs]).T

# start from random parameters
initial_guess = np.array(np.random.uniform(-10, 10, len(optimization_pairs)))


def fitness_func(ga_instance, solution, solution_idx):
    # map the parameters to between 0 and 1 and then to their respective bounds
    solution = sigmoid_bounding(solution, bounds)
    carrier, signal, noise = df.simulate_in_parallel(solution, *simulation_arrays[1:])
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[detector_ports]
    powers = powers[0] - powers[1]
    sensitivity = noise / np.abs(powers)

    # fitness/sensitivity relative to target fitness => fitness > 1 is better than voyager setup
    return np.sum(np.log10(target_sensitivity) - np.log10(sensitivity))

# genetic algorithm setup

fitness_function = fitness_func

num_generations = 3
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(optimization_pairs)

init_range_low = -10
init_range_high = 10

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

# document history
history_best = []
history_mean = []

# callback executed at the end of each generation
def on_generation(ga_instance):
	# compute fitness for each solution in the current population
	fitnesses = []
	for idx, sol in enumerate(ga_instance.population):
		# document fitnesses
		fitnesses.append(fitness_function(ga_instance, sol, idx))
	# store best and mean fitness for this generation
	history_best.append(float(np.max(fitnesses)))
	history_mean.append(float(np.mean(fitnesses)))

	# print progress: generation / total (percent) - best and mean fitness
	gen = ga_instance.generations_completed if hasattr(ga_instance, "generations_completed") else len(history_best)
	total = getattr(ga_instance, "num_generations", None)
	if total:
		pct = 100.0 * gen / total
		print(f"[GA] Generation {gen}/{total} ({pct:.1f}%) - best: {history_best[-1]:.6g}, mean: {history_mean[-1]:.6g}")
	else:
		print(f"[GA] Generation {gen} - best: {history_best[-1]:.6g}, mean: {history_mean[-1]:.6g}")


ga_instance = pygad.GA(num_generations=num_generations,
					   num_parents_mating=num_parents_mating,
					   fitness_func=fitness_function,
					   sol_per_pop=sol_per_pop,
					   num_genes=num_genes,
					   init_range_low=init_range_low,
					   init_range_high=init_range_high,
					   parent_selection_type=parent_selection_type,
					   keep_parents=keep_parents,
					   crossover_type=crossover_type,
					   mutation_type=mutation_type,
					   mutation_percent_genes=mutation_percent_genes,
					   on_generation=on_generation,
					   parallel_processing=('thread', 8))  # attach the callback

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = fitness_function(ga_instance, solution, solution_idx)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))


best_params = solution
losses = history_best

with open("voyager_optimization_parameters.json", "w") as f:
	json.dump(best_params.tolist(), f, indent=4)

with open("voyager_optimization_losses.json", "w") as f:
	json.dump(losses, f, indent=4)

with open("voyager_optimization_mean_fitness.json", "w") as f:
	json.dump(history_mean, f, indent=4)

plt.figure()
plt.plot(losses)
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.axhline(0, color="red", linestyle="--")
plt.grid()
plt.tight_layout()
plt.savefig("voyager_optimization_loss.png")


### Calculate the sensitivity of the best found setup ###
#-------------------------------------------------------#

update_setup(best_params, optimization_pairs, bounds, S)

carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports]
powers = powers[0] - powers[1]
sensitivity = noise / np.abs(powers)

plt.figure()
plt.plot(frequencies, sensitivity, label="Optimized Sensitivity")
plt.plot(frequencies, target_sensitivity, label="Target Sensitivity")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("voyager_optimization_sensitivity.png")
