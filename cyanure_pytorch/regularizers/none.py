import torch

from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.problem_param import ProblemParameters

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class NoRegul(Regularizer):

    def __init__(self, model : ProblemParameters) :
        super().__init__(model)     


    def prox(self, input : torch.Tensor, eta : float) -> torch.Tensor:    
        return input

    def eval_tensor(self,input : torch.Tensor) -> float:
        return 0

    def fenchel(self, grad1 : torch.Tensor, grad2 : torch.Tensor) -> float:
        return 0

    def provides_fenchel(self) -> bool:
        return False

    def print(self) -> None:
        logger.info("No regularization")