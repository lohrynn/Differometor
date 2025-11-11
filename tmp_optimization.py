import differometor as df
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup
import jax.numpy as jnp
from differometor.components import demodulate_signal_power
import numpy as np
import jax
import optax
import json
import torch
import time
import matplotlib.pyplot as plt


### Calculate the target sensitivity ###
#--------------------------------------#

# use a predefined Voyager setup with one noise detector and two signal detectors
S, component_property_pairs = voyager()

# set the frequency range
frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 100)

# run the simulation with the frequency as the changing parameter
carrier, signal, noise, detector_ports, *_ = df.run(S, [("f", "frequency")], frequencies)

# calculate the signal power at the detector ports
powers = demodulate_signal_power(carrier, signal)
powers = powers[detector_ports]

# calculate the signal power from the two signal detectors for balanced homodyne detection
powers = powers[0] - powers[1]

# calculate the sensitivity
target_sensitivity = noise / jnp.abs(powers)
target_loss = jnp.sum(jnp.log10(target_sensitivity))


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
initial_guess = jnp.array(np.random.uniform(-10, 10, len(optimization_pairs)))


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


def t2j(a):
    """Convert torch array to jax array."""
    return jax.dlpack.from_dlpack(a)


def j2t(a):
    """Convert jax array to torch array."""
    return torch.utils.dlpack.from_dlpack(a)


t = torch.rand(48)
t2 = torch.rand(48)
j = jax.random.uniform(jax.random.key(0), (48,))
j_100 = jax.random.uniform(jax.random.key(1), (20, 48))

v_obj = jax.jit(jax.vmap(objective_function, in_axes=0))

jitted_obj = jax.jit(objective_function)

for _ in range(5):
    start = time.time()
    _ = jax.vmap(objective_function)(j_100)
    end = time.time()
    print("Voyager vectorized objective function (x1, 100 pop):", (end - start) * 1000, "ms")

_ = jitted_obj(j)  # warmup

start = time.time()
for _ in range(20):
    _ = jitted_obj(j)
end = time.time()
mean_1 = (end - start)/20
print("Voyager objective function (x1):", mean_1 * 1000, "ms")


# collect results for plotting
pops = []
mean_pops = []

for pop in range(10, 61, 10):
    j_pop = jax.random.uniform(jax.random.key(pop), (pop,48))
    _ = v_obj(j_pop)  # warmup
    start = time.time()
    for _ in range(20):
        _ = v_obj(j_pop)
    end = time.time()
    mean_pop = (end - start)/20
    print(f"Voyager objective function for pop {pop} (x1):", mean_pop * 1000, "ms")
    pops.append(pop)
    mean_pops.append(mean_pop)

# compute ratios and fit least-squares line
pops_arr = np.array(pops)
ratios = np.array(mean_pops) / mean_1
slope, intercept = np.polyfit(pops_arr, ratios, 1)
fit_vals = slope * pops_arr + intercept

# plot results
plt.figure(figsize=(7,5))
plt.scatter(pops_arr, ratios, label="Measured ratio", color="C0")
plt.plot(pops_arr, fit_vals, label=f"Least-squares fit (slope={slope:.4f})", color="C1")
plt.xlabel("Population")
plt.ylabel("mean_pop / mean_1")
plt.title("Scaling of objective evaluation time with population")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("population_scaling.png", dpi=150)
plt.show()



