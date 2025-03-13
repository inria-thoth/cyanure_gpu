import torch
import math
from random import random
from math import floor

from cyanure_pytorch.losses.loss import Loss
from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.model_param import ModelParameters

from cyanure_pytorch.solvers.solver import Solver

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class IncrementalSolver(Solver):

    def __init__(self, loss : Loss, regul : Regularizer, param: ModelParameters, Li : torch.Tensor = None):
        super().__init__(loss, regul, param)

        self.non_uniform_sampling = param.non_uniform_sampling
        if (Li):
            self.Li = Li.clone()
        
        self.n = 0
        self.qi = None
        self.Ui = None
        self.Ki = None
        self.oldL = 0

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        
        if (self.Li is None):
            self.Li = self.loss.lipschitz_li(self.Li)
        self.n = self.Li.size(0)
        if (self.L == 0):
            self.qi = self.Li.clone()
            self.qi *= (1.0 / torch.sum(self.qi))
            Lmean = torch.mean(self.Li)
            Lmax = torch.max(self.Li)
            self.non_uniform_sampling = (self.non_uniform_sampling and Lmean <= 0.9 * Lmax)
            self.L = Lmean if self.non_uniform_sampling else Lmax
            if (self.minibatch > 1):
                self.heuristic_L(initial_weight)
            self.oldL = self.L
            if (self.non_uniform_sampling):
                self.init_nonu_sampling()

    def print(self) -> None: 
        logger.info("Incremental Solver ")
        if (self.non_uniform_sampling):
            logger.info("with non uniform sampling")
        else:
            logger.info("with uniform sampling")
        logger.info("Lipschitz constant: " + self.L)

    def init_nonu_sampling(self) -> None:
        self.Ui = torch.empty(self.n, dtype=self.qi.dtype)
        self.Ui.copy_(self.qi)
        self.Ui *= self.n / torch.sum(torch.abs(self.Ui))
        self.Ki = torch.zeros((self.n))
        overfull = [ii for ii, val in enumerate(self.Ui) if val > 1.0]
        underfull = [ii for ii, val in enumerate(self.Ui) if val < 1.0]
        while len(underfull) > 0 and len(overfull) > 0:
            indj = underfull.pop(0)  # Remove and get the first element
            indi = overfull.pop(0)   # Remove and get the first element
            
            self.Ki[indj] = indi
            self.Ui[indi] = self.Ui[indi] + self.Ui[indj] - 1.0
            
            if self.Ui[indi] < 1.0:
                underfull.append(indi)
            elif self.Ui[indi] > 1.0:
                overfull.append(indi)
        
    
    def nonu_sampling(self) -> int:
        x = (random() - 1) / float(2147483647)  # Assuming INT_MAX is 2147483647
        ind = int(floor(self.n * x)) + 1
        y = self.n * x + 1 - ind
        if y < self.Ui[ind - 1]:
            return ind - 1
        return int(self.Ki[ind - 1])
    
    def init_kappa_acceleration(self, initial_weight : torch.Tensor) -> float:
        self.solver_init(initial_weight)
        mu = self.regul.strong_convexity()
        return self.oldL / self.n - mu
    

    def heuristic_L(self, x : torch.Tensor) -> None:
        if (self.verbose):
            logger.info("Heuristic: Initial L=" + str(self.L))
        Lmax = self.L
        self.L /= self.minibatch;
        iteration = 0
        grad = None
        while (iteration <= math.log(self.minibatch) / math.log(2.0) and self.L < Lmax):
            tmp = x.clone()
            fx = self.loss.eval_random_minibatch(tmp, self.minibatch)
            grad = self.loss.grad_random_minibatch(tmp, grad, self.minibatch); # should do non uniform
            # compute grad and fx
            tmp.add(grad, -1.0 / self.L);
            ftmp = self.loss.eval_random_minibatch(tmp, self.minibatch)
            tmp2 = tmp.clone()
            tmp2.sub_(x)
            s1 = fx + grad.dot(tmp2);
            s2 = 0.5 * tmp2.nrm2sq();
            if (ftmp > s1 + self.L * s2):
                self.L = min(max(2.0 * self.L, (ftmp - s1) / s2), Lmax)
            iteration += 1
        if (self.verbose):
            logger.info(", Final L=" + str(self.L))
        