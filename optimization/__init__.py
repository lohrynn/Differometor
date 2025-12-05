"""Optimization package for Differometor.

Provides optimization algorithms and problem definitions.
"""

# Import protocols
from optimization.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)

# Import utilities
from optimization.config import create_parser
from optimization.utils import t2j, j2t, t2j_numpy, j2t_numpy

# Import algorithms
from optimization.algorithms.evox_pso import EvoxPSO
from optimization.algorithms.adam_gd import AdamGD
from optimization.algorithms.bayesian_optimization import BayesianOptimization
from optimization.algorithms.botorch_bo import BotorchBO

# Import problems
from optimization.problems.voyager_problem import VoyagerProblem

# Import benchamrking
from optimization.benchmark.benchmark import Benchmark, AlgorithmConfig


__all__ = [
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Algorithms
    "EvoxPSO",
    "AdamGD",
    "BayesianOptimization",
    "BotorchBO",
    # Problems
    "VoyagerProblem",
    # Utilities
    "create_parser",
    "t2j",
    "j2t",
    "t2j_numpy",
    "j2t_numpy",
    # Benchmarking
    "Benchmark",
    "AlgorithmConfig",
]
