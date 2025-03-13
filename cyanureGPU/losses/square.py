import torch

from cyanureGPU.losses.loss import LinearLossVec

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class SquareLoss(LinearLossVec):

    def __init__(self, data: torch.Tensor, y: torch.Tensor, intercept: bool):
        super().__init__(data, y, intercept)
        self.id = "SQUARE"

    def pre_compute(self, input: torch.Tensor) -> float:

        tmp = self.pred_tensor(input, None)

        tmp = torch.sub(tmp, self.labels)

        return tmp

    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> float:
        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = self.pred_tensor(input, None)
            grad1 = torch.sub(grad1, self.labels)
        else:
            grad1 = precompute
        return 0.5*torch.linalg.norm(grad1)**2/grad1.size(dim=0)

    def eval(self, input: torch.Tensor, i: int) -> float:
        res = self.labels[i] - self.pred(i, input)
        return 0.5*res*res

    def print(self) -> None:
        logger.info("Square Loss is used")

    def fenchel(self, input: torch.Tensor) -> float:
        return 0.5*torch.linalg.norm(input)**2/input.size(dim=0)+torch.dot(input, self.labels)/input.size(dim=0)

    def scal_grad(self, input: torch.Tensor, i: int) -> float:
        return self.pred(i, input, None) - self.labels[i]

    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None,
                     precompute: torch.Tensor = None) -> torch.Tensor:

        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = self.pred_tensor(input, None)
            grad1 = torch.sub(grad1, self.labels)
        else:
            grad1 = precompute

        return grad1

    def lipschitz_constant(self) -> float:
        return 1.0

    def get_dual_constraints(self, grad1: torch.Tensor) -> torch.Tensor:
        if (self.intercept):
            grad1 = grad1 - torch.mean(grad1)

        return grad1
