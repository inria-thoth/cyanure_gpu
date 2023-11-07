import torch

from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.problem_param import ProblemParameters

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class Ridge(Regularizer):

    def __init__(self, model : ProblemParameters) :
        super().__init__(model)

    def prox(self, input : torch.Tensor, eta : float) -> torch.Tensor:
        output = torch.clone(input)
        output = output * (1.0 / (1.0 + self.lambda_1 * eta))
        if (self.intercept):
            n = input.size(dim=0)
            output[n - 1] = input[n - 1]
        
        return output

    def eval_tensor(self, input : torch.Tensor) -> float:
        n = input.size(dim=0)
        res = torch.pow(torch.linalg.vector_norm(input), 2)
        return (0.5 * self.lambda_1 * (res - input[n - 1] * input[n - 1]) if self.intercept else 0.5 * self.lambda_1 * res)

    def fenchel(self, grad1 : torch.Tensor, grad2 : torch.Tensor) -> float:
        if (self.intercept and (abs(grad2[grad2.size(dim=0) - 1]) > 1e-6)):
            output = inf
        else:
            output = self.eval_tensor(grad2) / (self.lambda_1 * self.lambda_1)
        return output

    def print(self) -> None:
        logger.info(self.getName())

    def strong_convexity(self) -> float:
        return 0 if self.intercept else self.lambda_1

    def lazy_prox(self, input : torch.Tensor, output : torch.Tensor, indices : torch.Tensor, eta : float) -> None:
        scal = 1.0 / (1.0 + self.lambda_1 * eta)
        p = input.size(dim=0)
        r = indices.size(dim=0)
        for jj in range(r):
            output[indices[jj]] = scal * input[indices[jj]]
        if (self.intercept):
            output[p - 1] = input[p - 1]


    def is_lazy(self) -> bool:
        return True

    def getName(self) -> str:
        return "L2 regularization"