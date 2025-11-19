# HyperparameterVisualizer Documentation

## Overview

The `HyperparameterVisualizer` class provides powerful visualization tools for analyzing hyperparameter optimization results. It creates grid-based visualizations that allow you to compare the performance of different hyperparameter combinations at a glance.

## Features

### 1. Loss Curve Grids
Visualizes how the optimization loss evolves over iterations for different hyperparameter combinations.

### 2. Sensitivity Curve Grids
Visualizes the final sensitivity curves (physical performance) achieved by the best parameters found for each hyperparameter combination.

## Installation

The `HyperparameterVisualizer` is part of the `optimization.hpo` module:

```python
from optimization.hpo import HyperparameterVisualizer
from optimization import EvoxPSO, VoyagerProblem
```

## Basic Usage

### Initialize the Visualizer

```python
# Create problem and algorithm instances
problem = VoyagerProblem()
algorithm = EvoxPSO(problem=problem, batch_size=5)

# Define parameter space
param_space = {
    "pop_size": [10, 50, 100, 200],
    "n_generations": [10, 20, 50, 100],
}

# Create visualizer
visualizer = HyperparameterVisualizer(
    algorithm=algorithm,
    param_space=param_space,
    problem=problem
)
```

### Generate Loss Curve Grids

```python
# Create grid of loss curves
visualizer.visualize_loss_grids(
    figsize_per_subplot=(3, 2.5),  # Size of each subplot
    show_plot=False                 # Don't display, just save
)
```

This creates a grid where:
- **Rows** correspond to the first parameter (e.g., `pop_size`)
- **Columns** correspond to the second parameter (e.g., `n_generations`)
- Each subplot shows the loss curve for that parameter combination

### Generate Sensitivity Curve Grids

```python
# Create grid of sensitivity curves
visualizer.visualize_sensitivity_grids(
    figsize_per_subplot=(3.5, 2.5),
    show_plot=False
)
```

This creates a similar grid, but each subplot shows:
- The **target sensitivity** (baseline)
- The **optimized sensitivity** (achieved with best parameters)
- An **improvement metric** (Δ value in the corner)

## Advanced Usage

### Three or More Parameters

When you have more than 2 parameters, the visualizer creates **multiple grids**, one for each combination of the remaining parameters.

```python
param_space = {
    "pop_size": [100, 200, 300],       # Rows
    "n_generations": [1000, 2000],     # Columns
    "phi_p": [2.5, 5.0],               # Creates 2 separate grids
}

visualizer = HyperparameterVisualizer(
    algorithm=algorithm,
    param_space=param_space,
    problem=problem
)

# This will create 2 grids (one for phi_p=2.5, one for phi_p=5.0)
visualizer.visualize_loss_grids()
```

### Custom Save Paths

```python
# Save to a specific location
visualizer.visualize_loss_grids(
    save_path="./custom_path/my_loss_grid.png"
)

visualizer.visualize_sensitivity_grids(
    save_path="./custom_path/my_sensitivity_grid.png"
)
```

### Display Plots Interactively

```python
# Show plots in addition to saving them
visualizer.visualize_loss_grids(show_plot=True)
visualizer.visualize_sensitivity_grids(show_plot=True)
```

## Parameter Space Structure

The `param_space` dictionary should map parameter names to lists of values:

```python
param_space = {
    "param_name_1": [value1, value2, value3, ...],
    "param_name_2": [value1, value2, ...],
    # ... more parameters
}
```

### Requirements:
- Must have at least 2 parameters
- Each parameter must map to a non-empty list
- Parameter names should match those used when saving results

## File Path Resolution

The visualizer automatically finds result files based on:
1. The **algorithm name** (`algorithm.algorithm_str`)
2. The **problem name** (`problem._name`)
3. The **parameter combinations**

It looks for files in directories like:
```
./examples/{problem_name}/{algorithm_str}/gen{n_gen}_pop{pop_size}/
```

### Supported File Formats:
- **Loss files**: `*_losses*.json` - Contains array of loss values over iterations
- **Parameter files**: `*_parameters*.json` - Contains array of best parameter values

## Output Files

### Loss Grid
- **Filename**: `loss_grid_{row_param}_{col_param}[_{other_params}].png`
- **Location**: `{results_base_path}/`
- **Content**: Grid of loss curves with final loss values annotated

### Sensitivity Grid
- **Filename**: `sensitivity_grid_{row_param}_{col_param}[_{other_params}].png`
- **Location**: `{results_base_path}/`
- **Content**: Grid of sensitivity curves with improvement metrics

## Interpreting the Visualizations

### Loss Curves
- **Lower is better**: Negative values indicate better-than-baseline performance
- **Red dashed line**: Zero loss (baseline performance)
- **Final value**: Annotated in bottom-right corner of each subplot
- **Smooth curves**: Indicate stable convergence
- **Noisy curves**: May indicate instability or insufficient generations

### Sensitivity Curves
- **Gray dashed line**: Target sensitivity (baseline)
- **Blue line**: Optimized sensitivity
- **Green Δ**: Positive improvement (better than target)
- **Red Δ**: Negative improvement (worse than target)
- **Log scales**: Both axes use logarithmic scaling

## Example: Complete Workflow

```python
from optimization import EvoxPSO, VoyagerProblem
from optimization.hpo import HyperparameterVisualizer

# 1. Initialize problem and algorithm
problem = VoyagerProblem(name="voyager", n_frequencies=100)
algorithm = EvoxPSO(problem=problem, batch_size=5)

# 2. Define parameter space
param_space = {
    "pop_size": [10, 50, 100, 200],
    "n_generations": [10, 20, 50, 100],
}

# 3. Create visualizer
visualizer = HyperparameterVisualizer(
    algorithm=algorithm,
    param_space=param_space,
    problem=problem
)

# 4. Generate visualizations
print("Generating loss curves...")
visualizer.visualize_loss_grids()

print("Generating sensitivity curves...")
visualizer.visualize_sensitivity_grids()

print("Done! Check ./examples/voyager/evox_pso/ for results.")
```

## Troubleshooting

### "No data" appears in subplots
- Check that result files exist for those parameter combinations
- Verify the directory naming convention matches expected format
- Check file paths using `_get_result_file_path()` method

### "Error" appears in sensitivity plots
- Ensure the problem has a `calculate_sensitivity()` method
- Verify that parameter files contain valid parameter arrays
- Check that the problem can handle the loaded parameters

### Memory issues with large grids
- Reduce `figsize_per_subplot` to create smaller plots
- Process fewer parameter combinations at once
- Use `show_plot=False` to avoid memory overhead of displaying

## API Reference

### Constructor

```python
HyperparameterVisualizer(
    algorithm: OptimizationAlgorithm,
    param_space: dict[str, list],
    problem: ContinuousProblem,
    results_base_path: Optional[str] = None
)
```

**Parameters:**
- `algorithm`: The optimization algorithm instance
- `param_space`: Dictionary mapping parameter names to value lists
- `problem`: The problem being optimized
- `results_base_path`: Base directory for results (auto-detected if None)

### Methods

#### `visualize_loss_grids()`

```python
visualize_loss_grids(
    save_path: Optional[str] = None,
    figsize_per_subplot: tuple = (3, 2.5),
    show_plot: bool = False
) -> None
```

Generates grid visualizations of loss curves.

**Parameters:**
- `save_path`: Custom save path (auto-generated if None)
- `figsize_per_subplot`: Width and height of each subplot in inches
- `show_plot`: Whether to display the plot interactively

#### `visualize_sensitivity_grids()`

```python
visualize_sensitivity_grids(
    save_path: Optional[str] = None,
    figsize_per_subplot: tuple = (3, 2.5),
    show_plot: bool = False
) -> None
```

Generates grid visualizations of sensitivity curves.

**Parameters:**
- `save_path`: Custom save path (auto-generated if None)
- `figsize_per_subplot`: Width and height of each subplot in inches
- `show_plot`: Whether to display the plot interactively

**Note:** Requires problem to have `calculate_sensitivity()` method.

## See Also

- `VoyagerProblem` - Problem definition with sensitivity calculations
- `EvoxPSO` - Particle Swarm Optimization algorithm
- `HyperparameterOptimizer` - Automated hyperparameter optimization (future)
