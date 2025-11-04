import os

os.environ["MPLCONFIGDIR"] = "./tmp"  # TODO set this inside the env maybe
import jax
import torch
from jaxtyping import Array, Float
from typing import Callable
from abc import abstractmethod, ABC


def t2j(a: Array) -> jax.Array:
    """Convert torch array to jax array."""
    return jax.dlpack.from_dlpack(a)


def j2t(a: Array) -> torch.Tensor:
    """Convert jax array to torch array."""
    return torch.utils.dlpack.from_dlpack(a)


class ContinuousProblem(ABC):
    name: str

    objective_function: Callable[[Float[Array, "{self.n_params}"]], Float]

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def optimization_pairs(
        self,
    ) -> list[tuple]:
        pass

    @property
    def n_params(self) -> int:
        return len(self.optimization_pairs)

    def output_to_files(
        self,
        best_params: Float[Array, "{self.n_params}"] = None,
        losses: Float[Array, "iterations"] = None,
        population_losses: Float[Array, "iterations pop"] = None,
        algorithm_str: str = "",
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
    ) -> None:
        pass


class OptimizationAlgorithm(ABC):
    algorithm_str: str

    @abstractmethod
    def __init__(self, problem: ContinuousProblem, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, save_to_file: bool, *args, **kwargs):
        pass
