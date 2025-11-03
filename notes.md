# Notes
- Eine gute Initialisierung immer als richtlinie
- Stabilität bei Initialisierung charakterisieren
- Man kann noch eininges viel modularer machen! Aber erstmal verschiedene Algos ausprobieren von EvoX

# Modularization changelog
I am modularizing voyager_optimization to prepare for other algorithms and hyperparameter optimization. This should later be a subclass of a uifo optmization class which modularizes the more general optimization of a uifo setup. I think that starting with modularizing this more simple optimization is better.
### Created `class VoyagerOptimization`
- Moved much of the setup into `__init__()`
- Properties are:
  -  `optimization_pairs`
- Methods of the class:
  - `objective_function(optimized_parameters)` as a regular public attribute so it is pure and callable (wihtout self-reference and not as a property).
    - This function gets defined in `__init__()` to maintain pureness (otherwise it would use mutable `self.` variables). TODO review if this is necessary. 
  - `t2j(a)` for array conversion between `torch` and `jax`
  - `j2t(a)`
  - `output_to_files()` see below. TODO better name
- Variables are (crossed ones were removed as a class variable):
  - `self._setup`: Needed for `update_setup()` at file output.
  - `self._frequencies`: Needed for `df.run()` at file output
  - ~~`self._carrier`~~
  - ~~`self._signal`~~
  - ~~`self._noise`~~
  - ~~`self._detector_ports`~~
  - `self._target_sensitivity`
  - ~~`self._target_loss`~~
  - `self._optimization_pairs`: Needed for optimization algorithm
  - ~~`self._simulation_arrays`~~
  - ~~`self._detector_ports`~~
  - `self._bounds`: Needed for `update_setup()` for file output.
- File output is a new function `output_to_files()`
  - Outputs  best paramters and losses (over time) into json
  - Outputs Loss curve over time TODO option to plot over function calls.
  - Outputs the final setup's sensitivity curve.
  - Each argument can be left out which will just not plot/output the corresponding data.
  - Maybe looking for some magic methods
  - Arguments are:
    - `best_params`: Params for the final setup that get printed
    - `losses`: List of losses over time to plot
    - `population_losses`: List of lists of losses representing each pop. Graph could get messy. TODO: find a good plotting representation for high population cases.
  - Arguments TODO:
    - Naming scheme and directory. I'm thinking of creating an output directory and adding some string argument to the file name. Also add the date & time to the file name for better sorting. Is it necessary to create an argument to add date & time (defaulting to True)??

# Candidate Libraries
## PyGAD
- not that great
- Slow because of CPU
## EvoX (maybe also EvoGP, EvoRL)
- Seem great but are torch based
- Older jax-based EvalX version (0.9.0) exists

## evosax
- Seems good
- jax-based
- many algorithms (no PSO seemingly)

## numpyro
- jax-based, relies on jax' autodiff
- For bayesian regression

# Algos to try

## Gradient stuff

## Particle Swarm Optimization (PSO)
### Ressources
- [Getting started with PSO (EvoX)](https://evox.readthedocs.io/en/latest/examples/so-algorithm.html)
- [Custom Problems in EvoX](https://evox.readthedocs.io/en/stable/tutorial/tutorial_part5.html)

## Differential Evolution (DE)

## Bayesian Regression
