import torch

from cyanureGPU.regularizers.regularizer import Regularizer
from cyanureGPU.erm.param.problem_param import ProblemParameters

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class Lasso(Regularizer):

    def __init__(self, model: ProblemParameters):
        super().__init__(model)

        self.id = "L1"

    def fastSoftThrs(self, x: float) -> float:
        return x + 0.5*(torch.abs(x - self.lambda_1) - torch.abs(x + self.lambda_1))

    def prox(self, input: torch.Tensor, eta: float) -> torch.Tensor:
        output = input + 0.5 * (torch.abs(input - eta * self.lambda_1) - torch.abs(input + eta * self.lambda_1))

        if (self.intercept):
            if len(output.shape) == 1:
                n = input.size(dim=0)
                output[n - 1] = input[n - 1]
            else:
                p = input.size(dim=1)
                output[:, p - 1] = input[:, p-1]
        return output

    def eval_tensor(self, input: torch.Tensor) -> float:
        n = input.size(dim=0)
        res = torch.sum(torch.abs(input))
        return (self.lambda_1 * (res - torch.abs(input[n - 1])) if self.intercept else self.lambda_1 * res)

    def fenchel(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        indices = (torch.abs(grad2) == torch.max(torch.abs(grad2))).nonzero()[0]
        if len(indices) > 1:
            mm = torch.abs(grad2[indices[0], indices[1]])
        else:
            mm = torch.abs(grad2[indices])
        n = grad2.size(dim=0)
        if (mm > self.lambda_1):
            grad1 *= self.lambda_1 / mm
        return float("inf") if self.intercept and (torch.abs(grad2[n - 1]) > 1e-6) else 0, grad1, grad2

    def print(self) -> None:
        logger.info(self.getName())

    def lazy_prox(self, input: torch.Tensor, indices: torch.Tensor, eta: float) -> None:
        p = input.size(dim=0)
        # TODO output probablement faux
        # TODO plante surement en 1D
        output = torch.zeros(input.size())
        # Calculate the soft thresholding operation for all elements of the input tensor
        soft_thresholded = input + 0.5 * (torch.abs(input - self.lambda_1) - torch.abs(input + self.lambda_1))

        # Create a mask to select elements along the first dimension based on indices
        mask = torch.zeros(output)
        mask[indices, torch.arange(output.size(1))] = 1

        # Apply the mask to select elements along the first dimension and update them with soft thresholded values
        output.masked_scatter_(mask.bool(), soft_thresholded)
        if (self.intercept):
            if len(output.shape) == 1:
                n = input.size(dim=0)
                output[n - 1] = input[n - 1]
            else:
                p = input.size(dim=1)
                output[:, p - 1] = input[:, p-1]

        return output

    def is_lazy(self) -> bool:
        return True

    def getName(self) -> str:
        return "L1 regularization"
