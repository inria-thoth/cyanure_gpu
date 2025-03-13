import abc
import sys
import torch
import random

from typing import Tuple

from cyanureGPU.constants import EPSILON, DEVICE, TENSOR_TYPE

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class Loss:
    def __init__(self, data: torch.Tensor, y: torch.Tensor, intercept: bool) -> None:
        self.intercept = intercept
        self.scale_intercept = torch.tensor(1.0)
        self.input_data = data
        self.labels = y.to(TENSOR_TYPE)
        self.norms_vector = None
        self.grad_aux_compile = None

    # functions that should be implemented in derived classes (see also in protected section)
    @abc.abstractmethod
    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None) -> float:
        return

    @abc.abstractmethod
    def eval(self, input: torch.Tensor, i: int) -> float:
        return

    # should be safe if output=input1
    @abc.abstractmethod
    def add_grad(self, input: torch.Tensor, i: int, eta: float = 1.0) -> torch.Tensor:
        return

    @abc.abstractmethod
    def scal_grad(self, input: torch.Tensor, i: int) -> float:
        return

    @abc.abstractmethod
    def add_feature_tensor(self, input: torch.Tensor, input2: torch.Tensor, s: float) -> torch.Tensor:
        return

    @abc.abstractmethod
    def add_feature(self, i: int, s: float, input2: torch.Tensor) -> torch.Tensor:
        return

    @abc.abstractmethod
    def fenchel(self, input: torch.Tensor) -> float:
        return

    @abc.abstractmethod
    def print(self) -> None:
        return

    @abc.abstractmethod
    # virtual functions that may be reimplemented, if needed
    def double_add_grad(self, input1: torch.Tensor, input2: torch.Tensor,
                        i: int, eta1: float = 1.0, eta2: float = -1.0, dummy: float = 1.0) -> torch.Tensor:
        return

    def provides_fenchel(self) -> bool:
        return True

    def norms_data(self, norms: torch.Tensor) -> torch.Tensor:
        if (self.norms_vector is None or self.norms_tensor.size(dim=0) == 0):

            self.norms_tensor = torch.pow(torch.norm(self.input_data, dim=0), 2)

            if (self.intercept):
                if norms is not None:
                    self.norms_tensor = norms + torch.pow(self.scale_intercept, 2)
                else:
                    norms = torch.pow(self.scale_intercept, 2)

        norms = self.norms_tensor

        return norms

    def norms(self, ind: int) -> float:
        return self.norms[:, ind]

    def lipschitz(self) -> float:
        norms = None
        norms = self.norms_data(norms)
        return self.lipschitz_constant()*torch.max(norms)

    def lipschitz_li(self, Li: torch.Tensor) -> float:
        Li = self.norms_data(Li)
        Li = Li * self.lipschitz_constant()
        return Li

    def transpose(self) -> bool:
        return False

    def grad(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> torch.Tensor:
        tmp = self.get_grad_aux(input, matmul_result, precompute)
        gradient_size = tmp.size()
        if (len(gradient_size) > 1):
            n = gradient_size[1]
        else:
            n = gradient_size[0]
        return self.add_dual_pred_tensor(tmp, input, 1.0/n, 0)

    def eval_random_minibatch(self, input: torch.Tensor, minibatch: int) -> float:
        sum = 0
        n = self.n()
        for ii in range(minibatch):
            sum += self.eval(input, random.randint(0, sys.maxsize) % n)
        return (sum/minibatch)

    def grad_random_minibatch(self, input: torch.Tensor, grad: torch.Tensor, minibatch: int) -> torch.Tensor:
        n = self.n()
        for ii in range(minibatch):
            coef = 0.0 if ii == 0 else 1.0
            grad = self.add_grad(input, random.randint(0, sys.maxsize) % n, coef)
        return grad * (1.0/minibatch)

    def kappa(self) -> float:
        return 0

    def set_anchor_point(self, z: torch.Tensor) -> None:
        pass

    def get_anchor_point(self, z: torch.Tensor) -> None:
        pass

    def get_dual_variable(self, weight: torch.Tensor, grad_aux_result: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if grad_aux_result is None:
            grad1 = self.get_grad_aux(weight)
        else:
            grad1 = grad_aux_result
        grad1 = self.get_dual_constraints(grad1)
        gradient_size = grad1.size()
        if (len(gradient_size) > 1):
            n = gradient_size[1]
        else:
            n = gradient_size[0]
        return grad1, self.add_dual_pred_tensor(grad1, weight, 1.0/n, 0)

    # non-virtual function classes
    def n(self) -> int:
        return self.labels.size(0)

    def get_coordinates(self, ind: int) -> torch.Tensor:
        # TODO Wrong
        if (self.input_data.is_sparse):
            col = self.input_data[:, ind]
            indices = col

        return indices

    def set_intercept(self, initial_weight: torch.Tensor) -> torch.Tensor:
        self.scale_intercept = torch.sqrt(0.1 * torch.linalg.norm(self.input_data)**2 / self.input_data.size(dim=1))
        weight = torch.clone(initial_weight)
        if (len(weight.size()) == 1):
            weight[weight.size(0) - 1] /= self.scale_intercept
        else:
            weight[:, weight.size(1) - 1] /= self.scale_intercept

        return weight

    def reverse_intercept(self, weight: torch.Tensor) -> None:
        if (self.scale_intercept != 1.0):
            if (len(weight.size()) == 1):
                weight[weight.size(0) - 1] *= self.scale_intercept
            else:
                weight[:, weight.size(1) - 1] *= self.scale_intercept

        return weight

    def check_grad(self, input: torch.Tensor, output: torch.Tensor) -> None:
        output = torch.clone(input)
        x1 = torch.clone(input)
        x2 = torch.clone(input)
        for i in range(input.size(dim=1)):
            x1[i] -= 1e-7
            x2[i] += 1e-7
            output[i] = (self.eval_tensor(x2) - self.eval_tensor(x1)) / (2e-7)
            x1[i] += 1e-7
            x2[i] -= 1e-7

    @abc.abstractmethod
    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None) -> None:
        return

    @abc.abstractmethod
    def lipschitz_constant(self) -> float:
        return

    @abc.abstractmethod
    def get_dual_constraints(self, grad1: torch.Tensor) -> torch.Tensor:
        return

    @abc.abstractmethod
    def add_dual_pred_tensor(self, input: torch.Tensor, input2: torch.Tensor, a: float, b: float) -> torch.Tensor:
        return

    @abc.abstractmethod
    def add_dual_pred(self, ind: int, a: float, b: float) -> torch.Tensor:
        return

    @abc.abstractmethod
    def pred_tensor(self, input: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        return

    @abc.abstractmethod
    def pred(self, ind: int, input: torch.Tensor, input2: torch.Tensor) -> float:
        return


class LinearLossVec(Loss):

    def __init__(self, data: torch.Tensor, y: torch.Tensor, intercept: bool):
        super().__init__(data, y, intercept)

    def add_grad(self, input: torch.Tensor, i: int, a: float = 1.0) -> torch.Tensor:
        s = self.scal_grad(input, i, None)
        return self.add_dual_pred(i, a * s)

    def double_add_grad(self, input1: torch.Tensor, input2: torch.Tensor, i: int,
                        eta1: float = 1.0, eta2: float = -1.0, dummy: float = 1.0) -> torch.Tensor:
        res1 = self.scal_grad(input1, i, None)
        res2 = self.scal_grad(input2, i, None)
        if (res1 or res2):
            return self.add_dual_pred(i, eta1 * res1 + eta2 * res2)

    def add_feature(self, i: int, s: float, input2: torch.Tensor) -> torch.Tensor:
        return self.add_dual_pred(i, s, input2)

    def add_feature_tensor(self, input: torch.Tensor, input2: torch.Tensor, s: float) -> torch.Tensor:
        return self.add_dual_pred_tensor(input, input2, s, 1.0)

    @abc.abstractmethod
    def scal_grad(self, input: torch.Tensor, i: int) -> float:
        return

    def add_dual_pred_tensor(self, input: torch.Tensor, input2: torch.Tensor = None,
                             a: float = 1.0, b: float = 1.0) -> torch.Tensor:
        if (self.intercept):
            m = self.input_data.size(dim=0)
            output = torch.zeros((m + 1)).to(DEVICE)
            weight = self.get_w(input2)
            output[:-1] = a * torch.matmul(self.input_data, input) + weight * b
            output[-1] = self.scale_intercept * a * input.sum() + b * output[-1]
        else:
            if input2 is None:
                output = a * torch.matmul(self.input_data, input)
            else:
                output = a * torch.matmul(self.input_data, input) + b * input2
        return output

    def add_dual_pred(self, ind: int, a: float = 1.0, b: float = 1.0, input2: torch.Tensor = None) -> torch.Tensor:
        col = self.input_data[ind, :]
        m = self.input_data.size(dim=0)
        if input2 is None:
            input2 = torch.zeros((m))
        if (self.intercept):
            output = torch.zeros((m + 1))
            weight = self.get_w(input)
            col = a * weight + b * col
            output[m] = a * self.scale_intercept + b * output[m]
        else:
            output = a * col + b * input2

        return output

    def pred_tensor(self, input: torch.Tensor, input2: torch.Tensor = None) -> torch.Tensor:
        if (self.intercept):
            weight = self.get_w(input)
            output = torch.matmul(torch.transpose(self.input_data, 0, 1), weight)
            return output + (input[input.size(0) - 1] * self.scale_intercept)
        else:
            return torch.matmul(torch.transpose(self.input_data, 0, 1), input)

    def pred(self, ind: int, input: torch.Tensor, input2: torch.Tensor = None) -> float:
        col = self.input_data[ind, :]
        if (self.intercept):
            weight = self.get_w(input)
            return torch.dot(col, weight) + input[:, input.n() - 1] * self.scale_intercept
        else:
            return torch.dot(col, input)

    def get_w(self, input: torch.Tensor) -> torch.Tensor:
        n = input.size(0)
        return input[:n-1]


class ProximalPointLoss:

    def __init__(self, loss: Loss, z: torch.Tensor, kappa: float):
        self.anchor_point = z
        self.id = "PPA"
        self.kappa = kappa
        self.loss = loss

    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None,
                    precompute: torch.Tensor = None, fprox: torch.Tensor = None) -> float:
        tmp = torch.clone(input)
        tmp = tmp - self.anchor_point
        if fprox is None:
            return self.loss.eval_tensor(input) + 0.5 * self.kappa * torch.linalg.norm(tmp)**2
        else:
            return fprox + 0.5 * self.kappa * torch.linalg.norm(tmp)**2

    def eval(self, input: torch.Tensor, i: int) -> float:
        tmp = torch.clone(input)
        tmp.sub_(self.anchor_point)
        return self.loss.eval(input, i) + 0.5 * self.kappa * torch.linalg.norm(tmp)**2

    def grad(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> torch.Tensor:
        grad = self.loss.grad(input, matmul_result, precompute)
        grad = grad + input * self.kappa
        grad = grad + self.anchor_point * (-self.kappa)

        return grad

    def add_grad(self, input: torch.Tensor, i: int, eta: int = 1.0) -> torch.Tensor:
        grad = self.loss.add_grad(input, i, eta)
        grad = grad + input * self.kappa * eta
        grad = grad + self.anchor_point * (-self.kappa * eta)

        return grad

    def double_add_grad(self, input1: torch.Tensor, input2: torch.Tensor,
                        i: int, eta1: int = 1.0, eta2: int = -1.0, dummy: int = 1.0) -> torch.Tensor:
        output = self.loss.double_add_grad(input1, input2, i, eta1, eta2)
        if (dummy):
            output = output + input1 * dummy * self.kappa * eta1
            output = output + input2 * dummy * self.kappa * eta2
            if (abs(eta1+eta2) > EPSILON):
                output = output + self.anchor_point * (-self.kappa * dummy * (eta1 + eta2))

        return output

    def eval_random_minibatch(self, input: torch.Tensor, minibatch: int) -> float:
        sum_value = self.loss.eval_random_minibatch(input, minibatch)
        tmp = torch.clone(input)
        tmp.sub_(self.anchor_point)
        return sum_value + 0.5 * self.kappa * torch.linalg.norm(tmp)**2

    def grad_random_minibatch(self, input: torch.Tensor, minibatch: int) -> torch.Tensor:
        grad = self.loss.grad_random_minibatch(input, minibatch)
        grad = grad + input * self.kappa
        grad = grad + self.anchor_point * (-self.kappa)

        return grad

    def print(self) -> None:
        logger.info("Proximal point loss with")
        self.loss.print()

    def provides_fenchel(self) -> bool:
        return False

    def fenchel(self, input: torch.Tensor) -> float:
        return 0

    def lipschitz(self) -> float:
        return self.loss.lipschitz() + self.kappa

    def lipschitz_li(self, Li: torch.Tensor) -> torch.Tensor:
        Li = self.loss.lipschitz_li(Li)
        return Li + self.kappa

    def scal_grad(self, input: torch.Tensor, i: int) -> float:
        return self.loss.scal_grad(input, i, None)

    def add_feature_tensor(self, input: torch.Tensor, s: float) -> torch.Tensor:
        return self.loss.add_feature_tensor(input, s)

    def add_feature(self, i: int, s: float) -> torch.Tensor:
        return self.loss.add_feature(i, s)

    def kappa(self) -> float:
        return self.kappa

    def transpose(self) -> bool:
        return self.loss.transpose()

    @abc.abstractmethod
    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None) -> None:
        return

    @abc.abstractmethod
    def lipschitz_constant(self) -> int:
        return

    @abc.abstractmethod
    def get_dual_constraints(self, grad1: torch.Tensor):
        return
