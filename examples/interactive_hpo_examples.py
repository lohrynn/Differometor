"""
Interactive example for HyperparameterVisualizer.

This script demonstrates how to use the HyperparameterVisualizer interactively
to explore hyperparameter optimization results. It's designed to be run in a
Jupyter notebook or as a standalone script.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from optimization import EvoxPSO, VoyagerProblem
from optimization.hpo import HyperparameterVisualizer


def example_1_basic_usage():
    """Example 1: Basic usage with 2 parameters."""
    print("="*70)
    print("Example 1: Basic Usage with 2 Parameters")
    print("="*70)
    
    # Step 1: Initialize problem and algorithm
    problem = VoyagerProblem()
    algorithm = EvoxPSO(problem=problem, batch_size=5)
    
    # Step 2: Define parameter space
    # This should match the hyperparameters you actually ran optimization with
    param_space = {
        "pop_size": [10, 50, 100],
        "n_generations": [10, 20, 50],
    }
    
    print("\nParameter space:")
    for param, values in param_space.items():
        print(f"  {param}: {values}")
    
    # Step 3: Create visualizer
    visualizer = HyperparameterVisualizer(
        algorithm=algorithm,
        param_space=param_space,
        problem=problem
    )
    
    # Step 4: Generate visualizations
    print("\nGenerating loss curves grid...")
    visualizer.visualize_loss_grids(figsize_per_subplot=(3, 2.5))
    
    print("\nGenerating sensitivity curves grid...")
    visualizer.visualize_sensitivity_grids(figsize_per_subplot=(3.5, 2.5))
    
    print(f"\n✓ Results saved to: {visualizer.results_base_path}")


def example_2_custom_paths():
    """Example 2: Using custom save paths."""
    print("\n" + "="*70)
    print("Example 2: Custom Save Paths")
    print("="*70)
    
    problem = VoyagerProblem()
    algorithm = EvoxPSO(problem=problem, batch_size=5)
    
    param_space = {
        "pop_size": [10, 50],
        "n_generations": [10, 20],
    }
    
    visualizer = HyperparameterVisualizer(
        algorithm=algorithm,
        param_space=param_space,
        problem=problem
    )
    
    # Create custom output directory
    custom_output_dir = "./examples/custom_hpo_analysis"
    Path(custom_output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to custom directory: {custom_output_dir}")
    
    # Save with custom paths
    visualizer.visualize_loss_grids(
        save_path=f"{custom_output_dir}/my_loss_grid.png"
    )
    
    visualizer.visualize_sensitivity_grids(
        save_path=f"{custom_output_dir}/my_sensitivity_grid.png"
    )
    
    print(f"✓ Results saved to: {custom_output_dir}")


def example_3_analyzing_results():
    """Example 3: Programmatically analyzing results."""
    print("\n" + "="*70)
    print("Example 3: Programmatic Analysis of Results")
    print("="*70)
    
    problem = VoyagerProblem()
    algorithm = EvoxPSO(problem=problem, batch_size=5)
    
    param_space = {
        "pop_size": [10, 50, 100],
        "n_generations": [10, 20, 50],
    }
    
    visualizer = HyperparameterVisualizer(
        algorithm=algorithm,
        param_space=param_space,
        problem=problem
    )
    
    print("\nAnalyzing hyperparameter combinations...")
    print("-" * 70)
    
    # Iterate through all combinations and analyze
    best_loss = float('inf')
    best_combo = None
    
    for pop_size in param_space["pop_size"]:
        for n_gen in param_space["n_generations"]:
            combo = {"pop_size": pop_size, "n_generations": n_gen}
            
            # Load losses
            losses = visualizer._load_losses(combo)
            
            if losses is not None:
                final_loss = losses[-1]
                print(f"  pop_size={pop_size:3d}, n_generations={n_gen:3d}: "
                      f"final_loss={final_loss:8.3f}, iterations={len(losses)}")
                
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_combo = combo
            else:
                print(f"  pop_size={pop_size:3d}, n_generations={n_gen:3d}: "
                      f"No data available")
    
    if best_combo:
        print("-" * 70)
        print(f"\nBest hyperparameter combination:")
        for key, value in best_combo.items():
            print(f"  {key}: {value}")
        print(f"  Final loss: {best_loss:.3f}")


def example_4_comparing_ranges():
    """Example 4: Comparing different parameter ranges."""
    print("\n" + "="*70)
    print("Example 4: Comparing Different Parameter Ranges")
    print("="*70)
    
    problem = VoyagerProblem()
    algorithm = EvoxPSO(problem=problem, batch_size=5)
    
    # Define multiple parameter spaces to compare
    param_spaces = {
        "small_pop": {
            "pop_size": [10, 50],
            "n_generations": [10, 20, 50],
        },
        "large_pop": {
            "pop_size": [100, 200],
            "n_generations": [10, 20, 50],
        },
    }
    
    print("\nComparing parameter ranges...")
    
    for name, param_space in param_spaces.items():
        print(f"\n{name}:")
        for param, values in param_space.items():
            print(f"  {param}: {values}")
        
        visualizer = HyperparameterVisualizer(
            algorithm=algorithm,
            param_space=param_space,
            problem=problem
        )
        
        # Save with descriptive names
        output_dir = f"./examples/voyager/evox_pso"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        visualizer.visualize_loss_grids(
            save_path=f"{output_dir}/loss_grid_{name}.png"
        )
        
        print(f"  ✓ Saved loss grid for {name}")


def example_5_sensitivity_analysis():
    """Example 5: Detailed sensitivity analysis."""
    print("\n" + "="*70)
    print("Example 5: Detailed Sensitivity Analysis")
    print("="*70)
    
    problem = VoyagerProblem()
    algorithm = EvoxPSO(problem=problem, batch_size=5)
    
    param_space = {
        "pop_size": [50, 100],
        "n_generations": [20, 50],
    }
    
    visualizer = HyperparameterVisualizer(
        algorithm=algorithm,
        param_space=param_space,
        problem=problem
    )
    
    print("\nCalculating sensitivity improvements...")
    print("-" * 70)
    
    import jax.numpy as jnp
    
    for pop_size in param_space["pop_size"]:
        for n_gen in param_space["n_generations"]:
            combo = {"pop_size": pop_size, "n_generations": n_gen}
            
            # Load best parameters
            best_params = visualizer._load_parameters(combo)
            
            if best_params is not None:
                try:
                    # Calculate sensitivity
                    best_params_jax = jnp.array(best_params)
                    sensitivity = problem.calculate_sensitivity(best_params_jax)
                    sensitivity = np.array(sensitivity)
                    target_sensitivity = np.array(problem._target_sensitivity)
                    
                    # Calculate improvement in log space (geometric mean)
                    improvement = np.mean(
                        np.log10(target_sensitivity) - np.log10(sensitivity)
                    )
                    
                    status = "✓" if improvement > 0 else "✗"
                    print(f"  {status} pop_size={pop_size:3d}, n_generations={n_gen:3d}: "
                          f"improvement={improvement:+7.3f}")
                    
                except Exception as e:
                    print(f"  ✗ pop_size={pop_size:3d}, n_generations={n_gen:3d}: "
                          f"Error - {str(e)[:40]}")
            else:
                print(f"  - pop_size={pop_size:3d}, n_generations={n_gen:3d}: "
                      f"No data available")


def run_all_examples():
    """Run all examples in sequence."""
    print("\n" + "="*70)
    print("HyperparameterVisualizer Interactive Examples")
    print("="*70)
    
    examples = [
        example_1_basic_usage,
        example_2_custom_paths,
        example_3_analyzing_results,
        example_4_comparing_ranges,
        example_5_sensitivity_analysis,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_usage()
        elif example_num == "2":
            example_2_custom_paths()
        elif example_num == "3":
            example_3_analyzing_results()
        elif example_num == "4":
            example_4_comparing_ranges()
        elif example_num == "5":
            example_5_sensitivity_analysis()
        else:
            print(f"Unknown example number: {example_num}")
            print("Usage: python interactive_examples.py [1-5]")
            sys.exit(1)
    else:
        run_all_examples()
