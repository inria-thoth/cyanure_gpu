from typing import Tuple
import torch

from cyanureGPU.regularizers.regularizer import Regularizer
from cyanureGPU.erm.param.problem_param import ProblemParameters

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class RegMat(Regularizer):

    def __init__(self, regularizer: Regularizer, model: ProblemParameters, num_cols: int, transpose: bool):
        self.transpose = transpose
        self.num_cols = num_cols.type(torch.int32)
        self.regularizer_list = list()
        for i in range(self.num_cols):
            self.regularizer_list.append(type(regularizer)(model))

    def prox(self, input: torch.Tensor, eta: float) -> torch.Tensor:
        output = torch.clone(input)
        for i in range(self.num_cols):
            if self.transpose:
                colx = input[i, :]
            else:
                colx = input[:, i]
            coly = self.regularizer_list[i].prox(colx, eta)
            if self.transpose:
                output[i, :] = coly
            else:
                output[:, i] = coly

        return output

    def eval(self, input_tensor):
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        for i in range(self.num_cols):
            if self.transpose:
                col = input_tensor[i, :]
            else:
                col = input_tensor[:, i]
            value = self.regularizer_list[i].eval(col)
            sum_value += value

        return sum_value

    def fenchel(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        for i in range(self.num_cols):  # Assuming self.num_cols is defined somewhere
            if self.transpose:
                col1 = grad1[i, :]
                col2 = grad2[i, :]
            else:
                col1 = grad1[:, i]
                col2 = grad2[:, i]
            value = self.regularizer_list[i].fenchel(col1, col2)
            sum_value += value
            if self.transpose:
                grad1[i, :] = col1
                grad2[i, :] = col2
            else:
                grad1[:, i] = col1
                grad2[:, i] = col2

        return sum_value, grad1, grad2

    def provides_fenchel(self) -> bool:
        ok = True
        for i in range(self.num_cols):
            ok = ok and self.regularizer_list[i].provides_fenchel()

        return ok

    def print(self) -> None:
        logger.info("Regularization for matrices")
        self.regularizer_list[0].print()

    def lambda_1(self) -> float:
        return self.regularizer_list[0].lambda_1()

    def lazy_prox(self, input_tensor, indices, eta):
        output = torch.clone(input_tensor)

        for i in range(self.num_cols):
            if self.transpose:
                colx = input_tensor[i, :]
            else:
                colx = input_tensor[:, i]
            coly = self.regularizer_list[i].lazy_prox(colx, indices, eta)
            if self.transpose:
                output[i, :] = coly
            else:
                output[:, i] = coly

        return output

    def is_lazy(self) -> bool:
        return self.regularizer_list[0].is_lazy()


class RegVecToMat(Regularizer):

    def __init__(self, regularizer: Regularizer, model: ProblemParameters):
        super().__init__(model)
        parameter_tmp = model
        parameter_tmp.verbose = False
        self.regularizer = type(regularizer)(parameter_tmp)

    def prox(self, input: torch.Tensor, eta: float) -> torch.Tensor:
        weight, bias = self.get_wb(input)
        output = self.regularizer.prox(weight, eta)
        if self.intercept:
            output = torch.cat((output, torch.unsqueeze(bias, dim=1)), dim=1)
        return output

    def eval_tensor(self, input: torch.Tensor) -> float:
        weight, _ = self.get_wb(input)
        return self.regularizer.eval_tensor(torch.flatten(weight))

    def fenchel(self, grad1: torch.Tensor, grad2: torch.Tensor) -> float:
        weight, bias = self.get_wb(grad2)
        if self.intercept:
            bias_nrm_squ = torch.dot(bias, bias)
            if bias_nrm_squ > 1e-7:
                return float("inf"), grad1, grad2
        dual, grad1_flatten, weight_flatten = self.regularizer.fenchel(grad1.flatten(), weight.flatten())
        return dual, grad1_flatten.view(grad1.size()), weight_flatten.view(weight.size())

    def print(self) -> None:
        self.regularizer.print()

    def strong_convexity(self) -> float:
        return 0 if self.intercept else self.regularizer.strong_convexity()

    def lambda_1(self) -> float:
        return self.regularizer.lambda_1()

    def lazy_prox(self, input: torch.Tensor, indices: torch.Tensor, eta: float) -> None:
        weight, _ = self.get_wb(input)
        return self.regularizer.lazy_prox(weight, indices, eta)

    def is_lazy(self) -> bool:
        return self.regularizer.is_lazy()

    def get_wb(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = input.size(dim=1)
        if (self.intercept):
            weight = input[:, :p-1]
            bias = input[:, p-1]
        else:
            weight = input[:, :p]
            bias = None

        return weight, bias
