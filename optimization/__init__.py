"""Optimization package for Differometor.

Provides optimization algorithms and problem definitions.
"""

# Initialize environment variables first
import optimization._init_env  # noqa: F401

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
from optimization.algorithms.evolutionary.evox_pso import EvoxPSO
from optimization.algorithms.evolutionary.random_search import RandomSearch
from optimization.algorithms.gradient_based.adam_gd import AdamGD
from optimization.algorithms.gradient_based.sa_gd import SAGD
from optimization.algorithms.surrogate_based.botorch_bo import BotorchBO

# Import problems
from optimization.problems.voyager_problem import VoyagerProblem

# Import benchmarking
from optimization.benchmark.benchmark import Benchmark, AlgorithmConfig


__all__ = [
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Algorithms
    "EvoxPSO",
    "RandomSearch",
    "AdamGD",
    "SAGD",
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
