import torch
import math

from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.problem_param import ProblemParameters

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class Lasso(Regularizer):

    def __init__(self, model : ProblemParameters) :
        super().__init__(model)     

    def fastSoftThrs(self, x : float) -> float:
        return x + 0.5*(torch.abs(x - self.lambda_1) - torch.abs(x + self.lambda_1))
    
    def prox(self, input : torch.Tensor, eta : float) -> torch.Tensor:      

        output = input + 0.5 * (torch.abs(input - self.lambda_1) - torch.abs(input + self.lambda_1))
        if (self.intercept):
            n = input.size(dim=1)
            output[n - 1] = input[n - 1]
        return output

    def eval_tensor(self, input : torch.Tensor) -> float:
        n = input.size(dim=1)
        res = torch.sum(torch.abs(input))
        return (self.lambda_1 * (res - torch.abs(input[n - 1])) if self.intercept else self.lambda_1 * res)

    def fenchel(self, grad1 : torch.Tensor, grad2 : torch.Tensor) -> float:
        mm = grad2[torch.argmax(torch.abs(grad2))]
        n = input.size(dim=0)
        if (mm > self.lambda_1):
            grad1 *= self.lambda_1 / mm
        return float("inf") if self.intercept and (torch.abs(grad2[n - 1]) > 1e-6) else 0

    def print(self) -> None:
        logger.info(self.getName())

    def lazy_prox(self, input : torch.Tensor, indices : torch.Tensor, eta : float) -> None:
        p = input.size(dim=0)
        r = indices.size(dim=0)
        thrs = self.lambda_1 * eta
        output = torch.Tensor(input.size())

        for jj in range(r):
            output[indices[jj]] = self.fastSoftThrs(input[indices[jj]], thrs)
        if (self.intercept):
            output[p - 1] = input[p - 1]    

        return output

    def is_lazy(self) -> bool:
        return True

    def getName(self) -> str:
        return "L1 regularization"