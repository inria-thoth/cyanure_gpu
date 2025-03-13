import torch

from cyanureGPU.losses.loss import LinearLossVec

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class LogisticLoss(LinearLossVec):

    def __init__(self, data: torch.Tensor, y: torch.Tensor, intercept: bool):
        super().__init__(data, y, intercept)
        self.id = "LOGISTIC"

    def eval(self, input: torch.Tensor, i: int) -> float:
        res = self.labels[i] * self.pred(i, input)
        if res > 0:
            return torch.log(1.0 + torch.exp(-res))
        else:
            return torch.log(1.0 + torch.exp(res))

    def pre_compute(self, input: torch.Tensor) -> float:

        tmp = self.pred_tensor(input, None)

        tmp = torch.mul(tmp, self.labels)

        return tmp

    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> float:
        if precompute is None:
            if matmul_result is None:
                tmp = self.pred_tensor(input, None)
            else:
                tmp = matmul_result.clone()
            tmp = torch.mul(tmp, self.labels)
        else:
            tmp = precompute
        tmp = torch.log(1.0 + torch.exp(torch.neg(tmp)))
        return torch.sum(tmp) / (tmp.size(dim=0))

    def print(self) -> None:
        logger.info("Logistic Loss is used")

    def fenchel(self, input: torch.Tensor) -> float:

        n = input.size(dim=0)
        prod = torch.mul(self.labels, input)
        sum_vector = torch.special.xlogy(1.0+prod, 1.0+prod)+torch.special.xlogy(-prod, -prod)
        return torch.sum(sum_vector)/n

    def scal_grad(self, input: torch.Tensor, i: int) -> float:
        label = self.labels[i]
        ss = self.pred(i, input)
        s = -label/(1.0+torch.exp(label*ss))

        return s

    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None,
                     precompute: torch.Tensor = None) -> torch.Tensor:
        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = self.pred_tensor(input, None)
            grad1 = torch.mul(grad1, self.labels)
        else:
            grad1 = precompute

        grad1 = 1.0 / (torch.exp(grad1) + 1.0)
        grad1 = torch.mul(grad1, self.labels)
        grad1 = torch.neg(grad1)

        return grad1

    def lipschitz_constant(self) -> float:
        return 0.25

    def get_dual_constraints(self, grad1: torch.Tensor) -> torch.Tensor:
        if (self.intercept):
            grad1 = self.project_sft_binary(grad1, self.labels)

        return grad1

    def project_sft_binary(self, grad1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(grad1)
        if mean > 0:
            ztilde = grad1 + torch.where(y > 0, 1.0, 0.0)
            count = torch.sum(y > 0).item()
            xtilde = self.l1project(ztilde, count)
            grad1 = xtilde - torch.where(y > 0, 1.0, 0.0)
        else:
            ztilde = torch.where(y > 0, -grad1, -grad1 + 1.0)
            count = torch.sum(y <= 0).item()
            xtilde = self.l1project(ztilde, count)
            grad1 = torch.where(y > 0, -xtilde, -xtilde + 1.0)

        return grad1

    def l1project(self, input: torch.Tensor, thrs: float, simplex: bool = False) -> torch.Tensor:

        if simplex:
            output = torch.clamp(input, min=0)
        else:
            output = input.abs()

        norm1 = torch.sum(output)
        if norm1 <= thrs:
            return input if not simplex else output

        # Sort the input tensor in descending order
        sorted_output, _ = torch.sort(output, descending=True)

        # Calculate the cumulative sum
        cumulative_sum = torch.cumsum(sorted_output, dim=0) - thrs

        # Find rho, which is the largest index where the condition holds
        tmp = sorted_output * torch.arange(1, sorted_output.size(0) + 1, device=input.device) > cumulative_sum
        rho = torch.nonzero(tmp, as_tuple=True)[0].max()

        # Calculate the threshold lambda
        lambda_1 = cumulative_sum[rho] / (rho + 1)

        # Threshold the input tensor
        output = input.sign() * torch.clamp(output - lambda_1, min=0)

        return output
