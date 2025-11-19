# HyperparameterVisualizer Quick Reference

## Basic Usage

```python
from optimization import EvoxPSO, VoyagerProblem
from optimization.hpo import HyperparameterVisualizer

# 1. Setup
problem = VoyagerProblem()
algorithm = EvoxPSO(problem=problem, batch_size=5)

# 2. Define parameter space
param_space = {
    "pop_size": [10, 50, 100, 200],
    "n_generations": [10, 20, 50, 100],
}

# 3. Create visualizer
viz = HyperparameterVisualizer(algorithm, param_space, problem)

# 4. Generate plots
viz.visualize_loss_grids()
viz.visualize_sensitivity_grids()
```

## Parameter Space Rules

- **Minimum**: 2 parameters
- **Format**: `{"param_name": [value1, value2, ...]}`
- **Grid layout**: First 2 params → rows & columns
- **Additional params**: Create separate grids

## Common Patterns

### 2D Grid (Single)
```python
param_space = {
    "pop_size": [10, 50, 100],        # 3 rows
    "n_generations": [10, 20, 50],    # 3 columns
}
# Output: 1 grid (3×3)
```

### 3D Grid (Multiple)
```python
param_space = {
    "pop_size": [100, 200],           # 2 rows
    "n_generations": [1000, 2000],    # 2 columns
    "phi_p": [2.5, 5.0],              # 2 grids
}
# Output: 2 grids (each 2×2)
```

## Customization

### Figure Size
```python
viz.visualize_loss_grids(
    figsize_per_subplot=(4, 3)  # width, height in inches
)
```

### Save Location
```python
viz.visualize_loss_grids(
    save_path="./my_results/custom_name.png"
)
```

### Interactive Display
```python
viz.visualize_loss_grids(show_plot=True)
```

## Output Files

**Default location**: `./examples/{problem_name}/{algorithm_str}/`

**Filenames**:
- Loss grids: `loss_grid_{row_param}_{col_param}[_{other}].png`
- Sensitivity grids: `sensitivity_grid_{row_param}_{col_param}[_{other}].png`

## Interpreting Plots

### Loss Curves
- **Y-axis**: Loss value (lower is better)
- **Red dashed line**: Zero loss (baseline)
- **Annotation**: Final loss value
- **Missing data**: "No data" message

### Sensitivity Curves
- **Gray dashed**: Target sensitivity (baseline)
- **Blue solid**: Optimized sensitivity
- **Δ value**: Improvement metric
  - Green box → Better than target
  - Red box → Worse than target
- **Axes**: Both logarithmic

## Quick Commands

### Run Examples
```bash
# All examples
python examples/interactive_hpo_examples.py

# Specific example (1-5)
python examples/interactive_hpo_examples.py 3

# Quick test
python examples/quick_test_visualizer.py --size medium

# Main script
python examples/voyager_hpo_visualization.py
```

### CLI Options (quick_test_visualizer.py)
```bash
--size {small,medium,large}   # Grid size
--loss-only                   # Only loss grids
--sensitivity-only            # Only sensitivity grids
--show                        # Display interactively
--figsize WIDTH HEIGHT        # Custom subplot size
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No data" in plots | Check that result files exist in expected directories |
| Import errors | Ensure you're in the Differometor environment |
| Memory issues | Reduce `figsize_per_subplot` or grid size |
| "Error" in sensitivity plots | Verify problem has `calculate_sensitivity()` method |

## File Requirements

Expected directory structure:
```
examples/
  voyager/
    evox_pso/
      gen10_pop100/
        voyager_evox_pso_*_losses_gen10_pop100.json
        voyager_evox_pso_*_parameters_gen10_pop100.json
      gen20_pop100/
        ...
```

## Advanced Usage

### Programmatic Analysis
```python
# Access underlying data
losses = viz._load_losses({"pop_size": 100, "n_generations": 20})
params = viz._load_parameters({"pop_size": 100, "n_generations": 20})

# Analyze
if losses is not None:
    final_loss = losses[-1]
    print(f"Final loss: {final_loss}")
```

### Custom Base Path
```python
viz = HyperparameterVisualizer(
    algorithm, param_space, problem,
    results_base_path="./custom/path"
)
```

## Key Features

✓ Automatic file discovery  
✓ Grid-based comparison  
✓ Loss curve visualization  
✓ Sensitivity curve visualization  
✓ Improvement metrics  
✓ Missing data handling  
✓ Customizable appearance  
✓ Multiple parameter support  

## See Also

- **Full documentation**: `optimization/hpo/README.md`
- **Implementation details**: `IMPLEMENTATION_SUMMARY.md`
- **Example scripts**: `examples/` directory
