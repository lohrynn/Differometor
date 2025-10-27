import differometor as df
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
import matplotlib.pyplot as plt
import numpy as np
import jax
import optax
import json

import torch
from evox.algorithms import PSO
from evox.workflows import StdWorkflow, EvalMonitor
from evox.core import Problem

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
target_loss = noise / np.abs(powers)


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


def objective_function(optimized_parameters):
    # map the parameters to between 0 and 1 and then to their respective bounds
    optimized_parameters = sigmoid_bounding(optimized_parameters, bounds)
    carrier, signal, noise = df.simulate_in_parallel(optimized_parameters, *simulation_arrays[1:])
    powers = demodulate_signal_power(carrier, signal)
    powers = powers[detector_ports]
    powers = powers[0] - powers[1]
    sensitivity = noise / jnp.abs(powers)

    # loss relative to target loss => loss < 0 is better than voyager setup
    return jnp.sum(jnp.log10(sensitivity)) - target_loss

# PSO setup

num_generations = 100
pop_size = 100

# Maybe define a some universal names for those
lower_bound = -10
upper_bound = 10

algorithm = PSO(
    pop_size=pop_size,
    lb=lower_bound * torch.ones(len(optimization_pairs)),
    ub=upper_bound * torch.ones(len(optimization_pairs)),
)

def j2t(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def t2j(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x))

class VoyagerProblem(Problem):
    def __init__(self):
        super().__init__()
        self.vectorized_objective = jax.vmap(objective_function, in_axes=0)
        
    def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
        return j2t(self.vectorized_objective(t2j(pop)))
        
        
problem = VoyagerProblem()

monitor = EvalMonitor()

workflow = StdWorkflow(
    algorithm=algorithm,
    problem=problem,
    monitor=monitor,
)

# Perform optimization


for _ in range(num_generations): 
    workflow.step()
    


solution = monitor.topk_solutions
loss = monitor.topk_fitness
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {loss}".format(loss=loss))

prediction = objective_function(solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

with open("voyager_optimization_parameters.json", "w") as f:
	json.dump(solution.tolist(), f, indent=4)
 
losses = monitor.fit_history

with open("voyager_optimization_losses.json", "w") as f:
	json.dump(losses, f, indent=4)

# with open("voyager_optimization_mean_fitness.json", "w") as f:
# 	json.dump(losses, f, indent=4)

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

update_setup(solution, optimization_pairs, bounds, S)

carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports]
powers = powers[0] - powers[1]
sensitivity = noise / np.abs(powers)

plt.figure()
plt.plot(frequencies, sensitivity, label="Optimized Sensitivity")
plt.plot(frequencies, target_loss, label="Target Sensitivity")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Sensitivity [/sqrt(Hz)]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("voyager_optimization_sensitivity.png")
