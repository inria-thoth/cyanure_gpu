import abc
import torch

from cyanureGPU.erm.param.problem_param import ProblemParameters


class Regularizer:

    def __init__(self, model: ProblemParameters):
        self.intercept = model.intercept
        self.id = model.regul
        self.lambda_1 = model.lambda_1

    # should be able to do inplace with output=input
    @abc.abstractmethod
    def prox(self, input: torch.Tensor, eta: float) -> torch.Tensor:
        return

    @abc.abstractmethod
    def eval_tensor(self, input: torch.Tensor) -> float:
        return

    @abc.abstractmethod
    def fenchel(self, grad1: torch.Tensor, grad2: torch.Tensor) -> tuple[float, torch.Tensor, torch.Tensor]:
        return

    @abc.abstractmethod
    def print(self) -> None:
        return

    def is_lazy(self) -> bool:
        return False

    def lazy_prox(self, input: torch.Tensor, indices: torch.Tensor, eta: float) -> None:
        return None

    def provides_fenchel(self) -> bool:
        return True

    def id(self) -> str:
        return self.id

    def intercept(self) -> bool:
        return self.intercept

    def strong_convexity(self):
        return 0
