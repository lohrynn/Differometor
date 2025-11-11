"""Optimization package for Differometor.

Provides optimization algorithms and problem definitions.
"""

# Import protocols
from optimization.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
)

# Import algorithms
from optimization.algorithms.evox_pso import EvoxPSO
from optimization.algorithms.adam import AdamGD

# Import problems
from optimization.voyager.voyager_problem import VoyagerProblem

# Import utilities
from optimization.config import create_parser
from optimization.utils import t2j, j2t, t2j_numpy, j2t_numpy

__all__ = [
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    # Algorithms
    "EvoxPSO",
    "AdamGD",
    # Problems
    "VoyagerProblem",
    # Utilities
    "create_parser",
    "t2j",
    "j2t",
    "t2j_numpy",
    "j2t_numpy",
]
