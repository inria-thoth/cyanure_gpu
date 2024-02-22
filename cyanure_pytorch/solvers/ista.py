import torch
import math

from cyanure_pytorch.losses.loss import Loss
from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.model_param import ModelParameters

from cyanure_pytorch.solvers.solver import Solver

from cyanure_pytorch.constants import EPSILON

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class ISTA_Solver(Solver):

        global EPSILON
    
        def __init__(self, loss : Loss, regul : Regularizer, param: ModelParameters, Li : torch.Tensor = None):
            super().__init__(loss, regul, param)

            self.L = 0
            if (Li):
                self.Li = torch.clone(Li)
                self.L = torch.max(self.Li) / 100.0

        def solver_init(self, initial_weight: torch.Tensor) -> None:
            if (self.L == 0):
                self.Li = self.loss.lipschitz_li(self.Li)
                self.L = torch.max(self.Li) / 100.0
        
        def solver_aux(self, weight : torch.Tensor) -> torch.Tensor:
            iter = 1
            fx = self.loss.eval_tensor(weight)
            grad = self.loss.grad(weight)
            while (iter < self.max_iter_backtracking):
                
                tmp2 = torch.clone(weight)
                tmp2.sub_(grad, alpha=1.0/self.L)
                tmp = self.regul.prox(tmp2, 1.0 / self.L)
                fprox = self.loss.eval_tensor(tmp)
                tmp2 = torch.clone(tmp)
                tmp2.sub_(weight)
                if (fprox <= fx + torch.dot(torch.flatten(grad), torch.flatten(tmp2)) + 0.5 * self.L * torch.pow(torch.linalg.vector_norm(tmp2), 2) + EPSILON):
                    weight = torch.clone(tmp)
                    break
                self.L *= 1.5
                if (self.verbose):
                    logger.info("new value for L: " + str(self.L))
                iter += 1
                if (iter == self.max_iter_backtracking):
                    logger.warning("Warning: maximum number of backtracking iterations has been reached")
            
            return weight
        
        def print(self) -> None:
            logger.info("ISTA Solver")
        
        def init_kappa_acceleration(self, initial_weight : torch.Tensor) -> float:
            self.solver_init(initial_weight)
            return self.L
        



class FISTA_Solver(ISTA_Solver):
    def __init__(self, loss : Loss, regul : Regularizer, param: ModelParameters):
        super().__init__(loss, regul, param)

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        super().solver_init(initial_weight)
        self.t = 1.0
        self.labels = torch.clone(initial_weight)
    
    def solver_aux(self, weight : torch.Tensor) -> torch.Tensor:
        diff = torch.clone(weight)
        weight = super().solver_aux(self.labels)
        diff = diff - weight
        old_t = self.t
        self.t = (1.0 + torch.sqrt(1 + 4 * self.t * self.t)) / 2
        self.labels = self.labels + diff * (1.0 - old_t) / self.t
    
    def print(self) -> None:
        logger.info("FISTA Solver")