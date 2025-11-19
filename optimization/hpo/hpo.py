import os
import json
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from jaxtyping import Array, Float

from optimization import OptimizationAlgorithm, ContinuousProblem
from optimization.voyager.voyager_problem import VoyagerProblem


class HyperparameterVisualizer:
    """Generates visualizations of an algorithm's performance over different hyperparameter settings.

    This class creates grid visualizations for hyperparameter optimization results, showing
    how different hyperparameter combinations affect both loss curves and sensitivity curves.
    """

    def __init__(
        self,
        algorithm: OptimizationAlgorithm,
        param_space: dict[str, list],
        problem: ContinuousProblem,
        results_base_path: Optional[str] = None,
    ):
        """Initialize the HyperparameterVisualizer.

        Args:
            algorithm: The optimization algorithm used
            param_space: Dictionary mapping parameter names to lists of values
                        Example: {"pop_size": [100,200,300], "n_generations": [1000, 2000], "phi_p": [2.5, 5]}
            problem: The problem being optimized
            results_base_path: Base path where results are stored. If None, will be inferred
        """
        self.algorithm = algorithm
        self.param_space = param_space
        self.problem = problem

        # Infer base path if not provided
        if results_base_path is None:
            # Default to examples/{problem_name}/{algorithm_str}/
            problem_name = getattr(problem, "_name", "problem")
            algorithm_str = getattr(algorithm, "algorithm_str", "algorithm")
            self.results_base_path = f"./examples/{problem_name}/{algorithm_str}"
        else:
            self.results_base_path = results_base_path

        # Validate param_space
        if not param_space:
            raise ValueError("param_space cannot be empty")
        for key, values in param_space.items():
            if not isinstance(values, list) or len(values) == 0:
                raise ValueError(f"param_space['{key}'] must be a non-empty list")

    def _get_result_file_path(
        self, param_combination: dict, file_type: str
    ) -> Optional[str]:
        """Get the path to a result file for a given parameter combination.

        Args:
            param_combination: Dictionary of parameter names and values
            file_type: Type of file to look for ('losses', 'parameters', 'population_losses')

        Returns:
            Path to the most recent file matching the parameters, or None if not found
        """
        # Build directory path from param combination
        # Format: pop{pop_size}_gen{n_generations}_phi_p{phi_p}/
        param_str_parts = []
        for key, value in sorted(param_combination.items()):
            # Format the value nicely (handle floats and ints)
            if isinstance(value, float):
                value_str = str(value).replace(".", "_")
            else:
                value_str = str(value)
            param_str_parts.append(f"{key}{value_str}")

        # Check multiple possible directory naming conventions
        possible_dirs = []

        # Convention 1: gen{n_gen}_pop{pop_size} (standard format)
        if "n_generations" in param_combination and "pop_size" in param_combination:
            gen_pop_dir = f"gen{param_combination['n_generations']}_pop{param_combination['pop_size']}"
            possible_dirs.append(os.path.join(self.results_base_path, gen_pop_dir))

        # Convention 2: Directory with all parameters separated by underscores
        param_dir = "_".join(param_str_parts)
        possible_dirs.append(os.path.join(self.results_base_path, param_dir))

        # Try each possible directory
        for dir_path in possible_dirs:
            if not os.path.exists(dir_path):
                continue

            # Find all matching files
            matching_files = []
            for filename in os.listdir(dir_path):
                if file_type in filename and filename.endswith(".json"):
                    matching_files.append(os.path.join(dir_path, filename))

            if matching_files:
                # Return the most recent file
                return max(matching_files, key=os.path.getmtime)

        return None

    def _load_losses(self, param_combination: dict) -> Optional[np.ndarray]:
        """Load loss data for a given parameter combination.

        Args:
            param_combination: Dictionary of parameter names and values

        Returns:
            Array of losses, or None if file not found
        """
        file_path = self._get_result_file_path(param_combination, "losses")
        if file_path is None:
            return None

        try:
            with open(file_path, "r") as f:
                losses = json.load(f)
            return np.array(losses)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load losses from {file_path}: {e}")
            return None

    def _load_parameters(self, param_combination: dict) -> Optional[np.ndarray]:
        """Load best parameters for a given parameter combination.

        Args:
            param_combination: Dictionary of parameter names and values

        Returns:
            Array of parameters, or None if file not found
        """
        file_path = self._get_result_file_path(param_combination, "parameters")
        if file_path is None:
            return None

        try:
            with open(file_path, "r") as f:
                params = json.load(f)
            return np.array(params)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load parameters from {file_path}: {e}")
            return None

    def visualize_loss_grids(
        self,
        save_path: Optional[str] = None,
        figsize_per_subplot: tuple = (3, 2.5),
        show_plot: bool = False,
    ) -> None:
        """Generate grid visualizations of loss curves over the parameter space.

        Creates separate grids for each fixed value of parameters beyond the first two.
        For example, with param_space = {"pop_size": [100,200,300], "n_generations": [1000, 2000], "phi_p": [2.5, 5]},
        this creates two 3x2 grids (one for each phi_p value), where:
        - Each row corresponds to a pop_size
        - Each column corresponds to an n_generations value

        Args:
            save_path: Path to save the figure. If None, saves to results_base_path
            figsize_per_subplot: Size of each subplot in inches (width, height)
            show_plot: Whether to display the plot using plt.show()
        """
        # Get the parameter names and their values
        param_names = list(self.param_space.keys())
        if len(param_names) < 2:
            raise ValueError(
                "param_space must have at least 2 parameters for grid visualization"
            )

        # First two parameters define the grid dimensions
        row_param = param_names[0]  # e.g., pop_size
        col_param = param_names[1]  # e.g., n_generations
        row_values = self.param_space[row_param]
        col_values = self.param_space[col_param]

        # Remaining parameters create separate grids
        other_params = param_names[2:]

        # Generate all combinations of "other" parameters
        if other_params:
            from itertools import product

            other_combinations = list(
                product(*[self.param_space[p] for p in other_params])
            )
        else:
            other_combinations = [tuple()]  # Single grid if only 2 parameters

        # Create a figure for each combination of "other" parameters
        for other_combo in other_combinations:
            # Build title suffix from other parameters
            if other_params:
                title_suffix = ", ".join(
                    [f"{p}={v}" for p, v in zip(other_params, other_combo)]
                )
                filename_suffix = "_" + "_".join(
                    [
                        f"{p}{v}".replace(".", "_")
                        for p, v in zip(other_params, other_combo)
                    ]
                )
            else:
                title_suffix = ""
                filename_suffix = ""

            # Create grid
            n_rows = len(row_values)
            n_cols = len(col_values)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(
                    figsize_per_subplot[0] * n_cols,
                    figsize_per_subplot[1] * n_rows,
                ),
                squeeze=False,
            )

            # Populate grid
            for i, row_val in enumerate(row_values):
                for j, col_val in enumerate(col_values):
                    ax = axes[i, j]

                    # Build parameter combination
                    param_combo = {row_param: row_val, col_param: col_val}
                    for p, v in zip(other_params, other_combo):
                        param_combo[p] = v

                    # Load and plot losses
                    losses = self._load_losses(param_combo)

                    if losses is not None:
                        ax.plot(losses, linewidth=1.5)
                        ax.axhline(
                            0, color="red", linestyle="--", linewidth=1, alpha=0.7
                        )
                        ax.grid(True, alpha=0.3)
                        ax.set_xlabel("Iteration", fontsize=8)
                        ax.set_ylabel("Loss", fontsize=8)

                        # Add final loss value as text
                        final_loss = losses[-1]
                        ax.text(
                            0.98,
                            0.02,
                            f"Final: {final_loss:.2f}",
                            transform=ax.transAxes,
                            fontsize=7,
                            verticalalignment="bottom",
                            horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data",
                            transform=ax.transAxes,
                            fontsize=10,
                            verticalalignment="center",
                            horizontalalignment="center",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Add subplot title (only for top row)
                    if i == 0:
                        ax.set_title(
                            f"{col_param}={col_val}", fontsize=9, fontweight="bold"
                        )

                    # Add row labels (only for leftmost column)
                    if j == 0:
                        ax.set_ylabel(
                            f"{row_param}={row_val}\nLoss",
                            fontsize=8,
                            fontweight="bold",
                        )

            # Add overall title
            main_title = f"Loss Curves: {row_param} vs {col_param}"
            if title_suffix:
                main_title += f" ({title_suffix})"
            fig.suptitle(main_title, fontsize=12, fontweight="bold")

            plt.tight_layout()

            # Save figure
            if save_path is None:
                os.makedirs(self.results_base_path, exist_ok=True)
                save_path_final = os.path.join(
                    self.results_base_path,
                    f"loss_grid_{row_param}_{col_param}{filename_suffix}.png",
                )
            else:
                save_path_final = save_path

            plt.savefig(save_path_final, dpi=150, bbox_inches="tight")
            print(f"Saved loss grid to: {save_path_final}")

            if show_plot:
                plt.show()
            else:
                plt.close()

    def visualize_sensitivity_grids(
        self,
        save_path: Optional[str] = None,
        figsize_per_subplot: tuple = (3, 2.5),
        show_plot: bool = False,
    ) -> None:
        """Generate grid visualizations of best sensitivity curves over the parameter space.

        Creates separate grids for each fixed value of parameters beyond the first two.
        Each subplot shows the sensitivity curve for the best parameters found in that run,
        compared against the target sensitivity.

        Args:
            save_path: Path to save the figure. If None, saves to results_base_path
            figsize_per_subplot: Size of each subplot in inches (width, height)
            show_plot: Whether to display the plot using plt.show()
        """
        # Check if problem has calculate_sensitivity method
        if not isinstance(self.problem, VoyagerProblem):
            raise TypeError(
                "Problem must be a VoyagerProblem instance with calculate_sensitivity method"
            )

        # Get the parameter names and their values
        param_names = list(self.param_space.keys())
        if len(param_names) < 2:
            raise ValueError(
                "param_space must have at least 2 parameters for grid visualization"
            )

        # First two parameters define the grid dimensions
        row_param = param_names[0]
        col_param = param_names[1]
        row_values = self.param_space[row_param]
        col_values = self.param_space[col_param]

        # Remaining parameters create separate grids
        other_params = param_names[2:]

        # Generate all combinations of "other" parameters
        if other_params:
            from itertools import product

            other_combinations = list(
                product(*[self.param_space[p] for p in other_params])
            )
        else:
            other_combinations = [tuple()]

        # Get frequencies and target sensitivity from problem
        frequencies = np.array(self.problem.frequencies)
        target_sensitivity = np.array(self.problem._target_sensitivity)

        # Create a figure for each combination of "other" parameters
        for other_combo in other_combinations:
            # Build title suffix from other parameters
            if other_params:
                title_suffix = ", ".join(
                    [f"{p}={v}" for p, v in zip(other_params, other_combo)]
                )
                filename_suffix = "_" + "_".join(
                    [
                        f"{p}{v}".replace(".", "_")
                        for p, v in zip(other_params, other_combo)
                    ]
                )
            else:
                title_suffix = ""
                filename_suffix = ""

            # Create grid
            n_rows = len(row_values)
            n_cols = len(col_values)
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(
                    figsize_per_subplot[0] * n_cols,
                    figsize_per_subplot[1] * n_rows,
                ),
                squeeze=False,
            )

            # Populate grid
            for i, row_val in enumerate(row_values):
                for j, col_val in enumerate(col_values):
                    ax = axes[i, j]

                    # Build parameter combination
                    param_combo = {row_param: row_val, col_param: col_val}
                    for p, v in zip(other_params, other_combo):
                        param_combo[p] = v

                    # Load best parameters
                    best_params = self._load_parameters(param_combo)

                    if best_params is not None:
                        try:
                            # Calculate sensitivity for best parameters
                            import jax.numpy as jnp

                            best_params_jax = jnp.array(best_params)
                            sensitivity = self.problem.calculate_sensitivity(
                                best_params_jax
                            )
                            sensitivity = np.array(sensitivity)

                            # Plot target and optimized sensitivity
                            ax.plot(
                                frequencies,
                                target_sensitivity,
                                label="Target",
                                linestyle="--",
                                linewidth=1.5,
                                alpha=0.7,
                                color="gray",
                            )
                            ax.plot(
                                frequencies,
                                sensitivity,
                                label="Optimized",
                                linewidth=1.5,
                                color="blue",
                            )

                            ax.set_xscale("log")
                            ax.set_yscale("log")
                            ax.grid(True, alpha=0.3, which="both")
                            ax.set_xlabel("Frequency (Hz)", fontsize=8)
                            ax.set_ylabel("Sensitivity", fontsize=8)

                            # Calculate improvement ratio (mean over frequencies in log space)
                            improvement = np.mean(
                                np.log10(target_sensitivity) - np.log10(sensitivity)
                            )
                            color = "green" if improvement > 0 else "red"
                            ax.text(
                                0.98,
                                0.98,
                                f"Δ: {improvement:.2f}",
                                transform=ax.transAxes,
                                fontsize=7,
                                verticalalignment="top",
                                horizontalalignment="right",
                                bbox=dict(boxstyle="round", facecolor=color, alpha=0.3),
                            )

                        except Exception as e:
                            ax.text(
                                0.5,
                                0.5,
                                f"Error:\n{str(e)[:30]}",
                                transform=ax.transAxes,
                                fontsize=8,
                                verticalalignment="center",
                                horizontalalignment="center",
                            )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data",
                            transform=ax.transAxes,
                            fontsize=10,
                            verticalalignment="center",
                            horizontalalignment="center",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Add subplot title (only for top row)
                    if i == 0:
                        ax.set_title(
                            f"{col_param}={col_val}", fontsize=9, fontweight="bold"
                        )

                    # Add row labels (only for leftmost column)
                    if j == 0:
                        ax.set_ylabel(
                            f"{row_param}={row_val}\nSensitivity",
                            fontsize=8,
                            fontweight="bold",
                        )

                    # Add legend only to first subplot
                    if i == 0 and j == 0 and best_params is not None:
                        ax.legend(fontsize=7, loc="upper right")

            # Add overall title
            main_title = f"Sensitivity Curves: {row_param} vs {col_param}"
            if title_suffix:
                main_title += f" ({title_suffix})"
            fig.suptitle(main_title, fontsize=12, fontweight="bold")

            plt.tight_layout()

            # Save figure
            if save_path is None:
                os.makedirs(self.results_base_path, exist_ok=True)
                save_path_final = os.path.join(
                    self.results_base_path,
                    f"sensitivity_grid_{row_param}_{col_param}{filename_suffix}.png",
                )
            else:
                save_path_final = save_path

            plt.savefig(save_path_final, dpi=150, bbox_inches="tight")
            print(f"Saved sensitivity grid to: {save_path_final}")

            if show_plot:
                plt.show()
            else:
                plt.close()


class HyperparameterOptimizer:
    """Optimizes hyperparameters of a given optimization algorithm on a specified problem."""

    def __init__(
        self,
        algorithm: OptimizationAlgorithm,
        param_space: dict[str, list],
        problem: ContinuousProblem,
    ):
        self.algorithm = algorithm
        self.param_space = param_space
        self.problem = problem

    def optimize(self):
        """Finds the best hyperparameter settings for the algorithm on the problem."""
        pass
