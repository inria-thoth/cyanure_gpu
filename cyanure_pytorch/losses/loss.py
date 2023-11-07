import abc
import math
import torch

from typing import Tuple

class Loss:
    def __init__(self, data : torch.Tensor, y : torch.Tensor, intercept : bool) -> None:
        self.intercept = intercept
        self.scale_intercept = 1.0
        self.input_data = data
        self.labels = y
        self.norms_vector = None

    # functions that should be implemented in derived classes (see also in protected section)
    @abc.abstractmethod
    def eval_tensor(self, input : torch.Tensor) -> float:
        return

    @abc.abstractmethod
    def eval(self, input : torch.Tensor, i : int) -> float:
        return

    # should be safe if output=input1
    @abc.abstractmethod
    def add_grad(self, input : torch.Tensor, i : int, output : torch.Tensor, eta : float = 1.0) -> None:
        return 

    @abc.abstractmethod
    def scal_grad(self, input : torch.Tensor, i : int) -> float:
        return

    @abc.abstractmethod
    def add_feature_tensor(self, input : torch.Tensor, s : float)-> torch.Tensor:
        return

    @abc.abstractmethod
    def add_feature(self, i : int, s : float) -> torch.Tensor:
        return

    @abc.abstractmethod
    def fenchel(self, input : torch.Tensor) -> float:
        return

    @abc.abstractmethod
    def print(self) -> None:
        return 

    # virtual functions that may be reimplemented, if needed
    def double_add_grad(self, input1 : torch.Tensor, input2 : torch.Tensor, i : int, output : torch.Tensor, eta1 : float = 1.0, eta2 : float = -1.0, dummy : float = 1.0) -> None:
        add_grad(input1,i,output,eta1)
        add_grad(input2,i,output,eta2)

    def provides_fenchel(self) -> bool:
        return True

    def norms_data(self, norms : torch.Tensor) -> torch.Tensor:
        if (self.norms_vector is None or self.norms_tensor.size(dim=0) == 0):

            self.norms_tensor = torch.Tensor((self.input_data.size(dim=0)))

            for i in range(self.input_data.size(dim=0)):
                self.norms_tensor[i] = torch.pow(torch.linalg.vector_norm(self.input_data[:, i]), 2)

            if (self.intercept):
                norms = norms + math.pow(self.scale_intercept)
        
        return torch.clone(self.norms_tensor)

    def norms(self, ind : int) -> float:
        return self.norms[:, ind]

    def lipschitz(self) -> float:
        norms = self.norms_data(norms)
        return self.lipschitz_constant()*torch.max(norms)

    def lipschitz_li(self, Li : torch.Tensor) -> float:
        Li = self.norms_data(Li)
        Li = Li * self.lipschitz_constant()
        return Li
    
    def transpose(self) -> bool:
        return False

    def grad(self,input : torch.Tensor) -> torch.Tensor:
        tmp = self.get_grad_aux(input)
        return self.add_dual_pred_tensor(tmp,1.0/tmp.size(dim=0));
    
    def eval_random_minibatch(self, input: torch.Tensor, minibatch : int) -> float:
        sum = 0
        n = self.n()
        for ii in range(minibatch): 
            sum += self.eval(input,random.randint(0, sys.maxsize) % n)
        return (sum/minibatch)
    
    def grad_random_minibatch(self, input : torch.Tensor, grad : torch.Tensor, minibatch : int) -> torch.Tensor:
        n = self.n()
        for ii in range(minibatch):
            coef = 0.0 if ii == 0 else 1.0
            self.add_grad(input, random.randint(0, sys.maxsize) % n, grad, coef)       
        return grad * (1.0/minibatch)
    
    def kappa(self) -> float:
        return 0

    def set_anchor_point(self, z : torch.Tensor) -> None:
        pass

    def get_anchor_point(self, z : torch.Tensor) -> None:
        pass

    def get_dual_variable(self, input : torch.Tensor, grad2 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        grad1 = self.get_grad_aux(input)
        grad1 = self.get_dual_constraints(grad1)
        return grad1, self.add_dual_pred_tensor(grad1,1.0/grad1.size(dim=0))
    
    # non-virtual function classes
    def n(self) -> int:
        return self.labels.size(dim=1)

    def get_coordinates(ind : int, indices : torch.Tensor) -> None: 
        if (self.input_data.is_sparse):
            col = self.input_data.refCol[: , ind]
            indices = col
            
    def set_intercept(self, initial_weight : torch.Tensor, weight: torch.Tensor) -> None:
        self.scale_intercept = math.sqrt(0.1 * torch.pow(torch.linalg.vector_norm(self.input_data), 2) / self.input_data.size(dim=1));
        weight = torch.clone(initial_weight)
        weight[weight.size(dim=1) - 1] /= self.scale_intercept

    def reverse_intercept(self, weight: torch.Tensor) -> None:
        if (self.scale_intercept != 1.0):
            weight[weight.size(dim=1) - 1] *= self.scale_intercept

    def check_grad(self, input : torch.Tensor, output : torch.Tensor) -> None:
        output = torch.clone(input)
        x1 = torch.clone(input)
        x2 = torch.clone(input)
        for i in range(input.size(dim=1)):
            x1[ii] -= 1e-7
            x2[ii] += 1e-7
            output[ii]=(self.eval_tensor(x2)-self.eval_tensor(x1))/(2e-7);
            x1[ii] += 1e-7
            x2[ii] -= 1e-7

    @abc.abstractmethod
    def get_grad_aux(self, input : torch.Tensor)-> None:
        return

    @abc.abstractmethod
    def lipschitz_constant(self) -> float:
        return

    @abc.abstractmethod
    def get_dual_constraints(self, grad1 : torch.Tensor) -> torch.Tensor:
        return 

    def add_dual_pred_tensor(self, input : torch.Tensor, a : float = 1.0) -> torch.Tensor:
        weight = None
        if (self.intercept):
            m = self.input_data.size(dim=0)
            output = torch.Tensor((1, m + 1))
            weight = get_w(input, weight)
            weight = a * torch.matmul(self.input_data, input) + weight * b
            output[:, m] = self.scale_intercept * a * input.sum()
        else:
            output = a * torch.matmul(self.input_data, input)

        return output

    def add_dual_pred(self, ind : int, output : torch.Tensor, a : float = 1.0, b : float = 1.0) -> torch.Tensor:
        col = self.input_data[ind, :]
        if (self.intercept):
            m = self.input_data.size(dim=0)
            output.resize_((1, m + 1))
            weight = get_w(input, weight)
            col = a * weight +  b * col 
            output[:, m] = a * _scale_intercept + b * output[:, m]
        else:
            output.resize_((1, m))
            output = a * weight +  b * output 

        return output
        
    def pred_tensor(self, input : torch.Tensor) -> torch.Tensor:
        weight = None
        if (self.intercept): 
            weight = get_w(input, weight)
            output = torch.matmul(torch.transpose(self.input_data, 0, 1), weight)
            return output  + (input[input.size(dim=1) - 1] * self.scale_intercept)
        else:
            return torch.matmul(torch.transpose(self.input_data, 0, 1), input)

    def pred(ind : int, input: torch.Tensor) -> float:
        col = None
        weight = None
        col = self.input_data[ind, :]
        if (self.intercept):
            get_w(input, weight)
            return torch.dot(col, weight) + input[:, input.n() - 1] * self.scale_intercept
        else:
            return torch.dot(col, input)

    def get_w(input : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:
        n = input.n()
        if (len(weight.size()) == 1):
            weight = input[:n-1]
        else:
            weight = input[:, :n-1]

        return weight


class ProximalPointLoss:

    def __init__(self, loss : Loss, z : torch.Tensor, kappa : float): 
        self.anchor_point = z
        self.id="PPA"
        self.kappa = kappa
        self.loss = loss

    def eval_tensor(self, input : torch.Tensor) -> float:
        tmp = torch.clone(input);
        tmp = torch.sub(tmp, self.anchor_point)
        return self.loss.eval_tensor(input)+ 0.5 * self.kappa * torch.pow(torch.linalg.vector_norm(tmp), 2)

    def eval(self, input : torch.Tensor, i: int) -> float:
        tmp = torch.clone(input);
        tmp = torch.sub(tmp, self.anchor_point)
        return self.loss.eval(input, i) + 0.5 * self.kappa * torch.pow(torch.linalg.vector_norm(tmp), 2)

    def grad(self, input : torch.Tensor) -> torch.Tensor:
        grad = self.loss.grad(input)
        grad = grad + input * self.kappa
        grad = grad + self.anchor_point * (-self.kappa)

        return grad

    def add_grad(self, input : torch.Tensor, i : int, eta : int = 1.0) -> torch.Tensor: 
        grad = self.loss.add_grad(input, i, eta)
        grad = grad + input * self.kappa * eta
        grad = grad + self.anchor_point * (-self.kappa * eta)

        return grad

    def double_add_grad(self, input1 : torch.Tensor, input2 : torch.Tensor, i : int, eta1 : int = 1.0, eta2 : int = -1.0, dummy : int = 1.0) -> torch.Tensor:
        output = self.loss.double_add_grad(input1, input2, i, eta1, eta2)
        if (dummy):
            output = output + input1 * dummy * self.kappa * eta1
            output = output + input2 * dummy * self.kappa * eta2
            if (abs(eta1+eta2) > EPSILON):
                output = output + self.anchor_point * (-self.kappa * dummy * (eta1 + eta2))

        return output
    
    def eval_random_minibatch(self, input : torch.Tensor, minibatch : int) -> float:
        sum_value = self.loss.eval_random_minibatch(input, minibatch)
        tmp = torch.clone(input);
        tmp = torch.sub(tmp, self.anchor_point)
        return sum_value + 0.5 * self.kappa * torch.pow(torch.linalg.vector_norm(tmp), 2)

    def grad_random_minibatch(self, input : torch.Tensor, minibatch : int) -> torch.Tensor:
        grad = self.loss.grad_random_minibatch(input, minibatch);
        grad = grad + input * self.kappa
        grad = grad + self.anchor_point * (-self.kappa)

        return grad

    def print(self) -> None:
        logger.info("Proximal point loss with")
        self.loss.print()
    
    def provides_fenchel(self) -> bool:
        return False

    def fenchel(self, input : torch.Tensor) -> float:
        return 0

    def lipschitz(self) -> float:
        return self.loss.lipschitz() + self.kappa

    def lipschitz_li(self, Li : torch.Tensor) -> torch.Tensor:
        Li = self.loss.lipschitz_li(Li)
        return Li + self.kappa

    def scal_grad(self, input : torch.Tensor, i : int) -> float:
        return self.loss.scal_grad(input,i)

    def add_feature_tensor(self, input : torch.Tensor, s : float) -> torch.Tensor:
        return self.loss.add_feature_tensor(input, s)

    def add_feature(self, i : int, s : float) -> torch.Tensor:
        return self.loss.add_feature(i, s)

    def kappa(self) -> float: 
        return self.kappa

    def transpose(self) -> bool: 
        return self.loss.transpose()

    @abc.abstractmethod
    def get_grad_aux(self, input : torch.Tensor, grad1 : torch.Tensor) -> None:
        return

    @abc.abstractmethod
    def lipschitz_constant(self) -> int:
         return

    @abc.abstractmethod
    def get_dual_constraints(self, grad1 : torch.Tensor):
        return
