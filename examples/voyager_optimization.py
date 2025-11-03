import os

os.environ["MPLCONFIGDIR"] = "./tmp" # TODO set this inside the env maybe by: export MPLCONFIGDIR=/mnt/lustre/work/krenn/klz895/Differometor/tmp
from datetime import datetime
import differometor as df
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup
import jax.numpy as jnp
import jax
from jaxtyping import Array, Float
from differometor.components import demodulate_signal_power
import matplotlib.pyplot as plt
import numpy as np
import optax
import torch
import json


class VoyagerProblem:
    def __init__(self, name: str = "voyager"):
        self._name = name.lstrip('_')
        ### Calculate the target sensitivity ###
        # --------------------------------------#
        # use a predefined Voyager setup with one noise detector and two signal detectors
        self._setup, component_property_pairs = voyager()

        # set the frequency range
        self._frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), 100)

        # run the simulation with the frequency as the changing parameter
        carrier, signal, noise, detector_ports, *_ = df.run(
            self._setup, [("f", "frequency")], self._frequencies
        )

        # calculate the signal power at the detector ports
        powers = demodulate_signal_power(carrier, signal)
        powers = powers[detector_ports]

        # calculate the signal power from the two signal detectors for balanced homodyne detection
        powers = powers[0] - powers[1]

        # calculate the sensitivity
        self._target_sensitivity = noise / jnp.abs(powers)
        target_loss = jnp.sum(jnp.log10(self._target_sensitivity))

        ### Start from random parameters and optimize the sensitivity ###
        # ---------------------------------------------------------------#

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
        optimized_properties = [
            "reflectivity",
            "tuning",
            "db",
            "angle",
            "power",
            "mass",
            "length",
            "phase",
        ]
        self._optimization_pairs = []
        for pair in component_property_pairs:
            if pair[1] in optimized_properties:
                self._optimization_pairs.append(pair)

        # build the setup once and then reuse it during the optimization
        simulation_arrays, detector_ports, *_ = df.run_build_step(
            self._setup,
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )

        # calculate the bounds for the properties to be optimized
        self._bounds = np.array(
            [
                [property_bounds[pair[1]][0], property_bounds[pair[1]][1]]
                for pair in self._optimization_pairs
            ]
        ).T

        # abstract for pure objective_function
        bounds = self._bounds

        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            # map the parameters to between 0 and 1 and then to their respective bounds
            optimized_parameters = sigmoid_bounding(optimized_parameters, bounds)
            carrier, signal, noise = df.simulate_in_parallel(
                optimized_parameters, *simulation_arrays[1:]
            )
            powers = demodulate_signal_power(carrier, signal)
            powers = powers[detector_ports]
            powers = powers[0] - powers[1]
            sensitivity = noise / jnp.abs(powers)

            # loss relative to target loss => loss < 0 is better than voyager setup
            return jnp.sum(jnp.log10(sensitivity)) - target_loss

        self.objective_function = objective_function

    @property
    def optimization_pairs(
        self,
    ) -> list[tuple]:  # TODO: Find out what's inside that tuple
        return self._optimization_pairs

    @property
    def n_params(self) -> int:
        """Number of parameters to be optimized. Is equal to len(self.optimization_pairs)."""
        return len(self._optimization_pairs)

    def t2j(a):
        """Convert torch array to jax array."""
        return jax.dlpack.from_dlpack(a)
    
    def j2t(a):
        """Convert jax array to torch array."""
        return torch.utils.dlpack.from_dlpack(a)

    def output_to_files(
        self,
        best_params: Float[Array, "{self.n_params}"] = None,
        losses: Float[Array, "iterations"] = None,
        population_losses: Float[Array, "iterations pop"] = None,
        algorithm_str: str = "",
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
    ) -> None:

        # Prepare strings and timestamp
        algorithm_str = f"_{algorithm_str.strip('_')}" if algorithm_str != "" else ""
        hyper_param_str = f"_{hyper_param_str.strip('_')}" if hyper_param_str != "" else ""
        timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M")

        
        # Create output directory
        output_path = os.path.join(
            f"./examples/{self._name}/{algorithm_str.strip('_')}",
            hyper_param_str.strip('_'), # directory should not have leading underscore
        )
        os.makedirs(output_path, exist_ok=True)

        # Send info to user
        print(f"Output directory: {output_path}")
        
        # Determine file name prefix and suffix
        file_prefix = f"{self._name}{algorithm_str}{timestamp}"
        file_suffix = hyper_param_str if hyper_param_str_in_filename else ""

        # Output best parameters to JSON
        with open(
            os.path.join(
                output_path, f"{file_prefix}_parameters{file_suffix}.json"
            ),
            "w",
        ) as f:
            json.dump(best_params.tolist(), f, indent=4)

        # Output historical losses to JSON
        with open(
            os.path.join(
                output_path, f"{file_prefix}_losses{file_suffix}.json"
            ),
            "w",
        ) as f:
            json.dump(losses.tolist(), f, indent=4)


        is_genetic = population_losses is not None

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Generation" if is_genetic else "Iteration")
        plt.ylabel("Best losses" if is_genetic else "Loss")
        plt.axhline(0, color="red", linestyle="--")
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_path, f"{file_prefix}_losses{file_suffix}.png"
            )
        )

        if population_losses is not None:
            
            plt.figure()
            plt.plot(population_losses)
            plt.xlabel("Generation")
            plt.ylabel("All losses")
            plt.axhline(0, color="red", linestyle="--")
            plt.grid()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_path, f"{file_prefix}_population_losses{file_suffix}.png"
                )
            )

        ### Calculate the sensitivity of the best found setup ###
        # -------------------------------------------------------#

        update_setup(best_params, self._optimization_pairs, self._bounds, self._setup)

        carrier, signal, noise, detector_ports, *_ = df.run(
            self._setup, [("f", "frequency")], self._frequencies
        )
        powers = demodulate_signal_power(carrier, signal)
        powers = powers[detector_ports]
        powers = powers[0] - powers[1]
        sensitivity = noise / jnp.abs(powers)

        plt.figure()
        plt.plot(self._frequencies, sensitivity, label="Optimized Sensitivity")
        plt.plot(self._frequencies, self._target_sensitivity, label="Target Sensitivity")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Sensitivity [/sqrt(Hz)]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_path, f"{file_prefix}_sensitivity{file_suffix}.png"
            )
        )


class AdamGD:
    def __init__(
        self, problem: VoyagerProblem, max_iterations: int = 50000, patience: int = 1000
    ):
        """Initialize gradient descent

        Args:
            problem (VoyagerProblem): The problem being optimized
            max_iterations (int): Maximum number of iterations. Defaults to 50,000
            patience (int): Stop if no improvement after this many iterations. Defaults to
        """
        self._problem = problem
        self.max_iterations = max_iterations
        self.patience = patience
        self._best_params = jnp.array(
            np.random.uniform(-10, 10, self._problem.n_params)
        )
        self._losses = []
        self._best_loss = 1e10

        self._grad_fn = jax.jit(jax.value_and_grad(self._problem.objective_function))

    def run(self, save_to_file: bool = True, learning_rate: Float = 0.1, **adam_kwargs):
        """Run optimization with Adam.

        Args:
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.1
            **adam_kwargs: Additional keyword arguments passed to optax.adam()
                          (Default: b1=0.9, b2=0.999, eps=1e-8, eps_root=0.0, nesterov=False)
        """
        # warmup the function to compile it
        _ = self._grad_fn(self._best_params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(self._best_params)

        params, no_improve_count, losses = self._best_params, 0, []

        for i in range(self.max_iterations):
            loss, grads = self._grad_fn(params)

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss}")

            if loss < self._best_loss - 1e-4:
                self._best_loss, self._best_params, no_improve_count = loss, params, 0
                print(f"Iteration {i}: New best loss = {loss}")
            else:
                no_improve_count += 1

            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))

            # if the loss has not improved (< best_loss - 1e-4) over 1000 iterations, stop the optimization
            if no_improve_count > self.patience:
                break

        self._losses = jnp.array(losses)
        
        
        if save_to_file:
            self._problem.output_to_files(best_params=self._best_params, losses=self._losses, algorithm_str="adam", hyper_param_str=f"lr{learning_rate}") # TODO conditionally add more hyperparameters to string

vp = VoyagerProblem()

optimizer = AdamGD(vp, max_iterations=200)

optimizer.run()

