import abc

import torch
import numpy
import time

from cyanure_pytorch.erm.param.model_param import ModelParameters

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class Solver:

    global EPSILON
    NUMBER_OPTIM_PROCESS_INFO = 6

    def __init__(self, loss, regul, param: ModelParameters):
        self.verbose = param.verbose
        self.duality_gap_interval = max(param.duality_gap_interval, 1)
        self.tol = param.tol
        self.max_iter = param.max_iter
        self.max_iter_backtracking = param.max_iter_backtracking
        self.best_dual = -float("Inf")
        self.best_primal = float("Inf")
        self.optim_info = torch.empty((self.NUMBER_OPTIM_PROCESS_INFO, max(int(param.max_iter / self.duality_gap_interval), 1)))
        self.L = 0
        self.Li = None
        self.minibatch = param.minibatch
        self.previous_weight = None
        self.loss = loss
        self.regul = regul
        self.elapsed_time = 0
        self.duality = self.loss.provides_fenchel() and self.regul.provides_fenchel()

    def solve(self, initial_weight : torch.Tensor, weight : torch.Tensor) -> torch.Tensor:

        initial_time = time.time()
        weight = torch.clone(initial_weight)

        if (not self.duality and self.max_iter > 1):
            self.previous_weight = torch.Tensor(initial_weight)

        self.solver_init(initial_weight)
        if (self.verbose):
            logger.info("*********************************")
            self.loss.print()
            self.regul.print()

        for it in range(1, self.max_iter + 1):
            if ((it % self.duality_gap_interval) == 0):
                if (self.test_stopping_criterion(weight, it)):
                    break
            weight = self.solver_aux(weight)
            self.elapsed_time = time.time() - initial_time
        if (self.verbose):
            logger.info("This is the elapsed time: " + str(self.elapsed_time))
        if (self.best_primal != float("Inf")):
            weight = torch.clone(self.best_weight)

        return weight
    
    def get_optim_info(self) -> None:
        count = 0
        for index in range(self.optim_info.size(dim=1)):
            if (self.optim_info[0, index] != 0):
                count+=1
        if (count > 0):
            optim = torch.Tensor(1, self.NUMBER_OPTIM_PROCESS_INFO, count)

        for index in range(count):
            for inner_index in range (self.NUMBER_OPTIM_PROCESS_INFO):
                optim[0, inner_index, index] = self.optim_info[inner_index, index]

        if (count > 0):
            return optim
        else:
            return self.optim_info

    def eval(self, x : torch.Tensor) -> None:
        self.test_stopping_criterion(x, 1)
        self.optim_info[5, 0] = 0

    @abc.abstractmethod
    def set_dual_variable(self, initial_dual : float):
        return

    @abc.abstractmethod
    def save_state(self):
        return

    @abc.abstractmethod
    def restore_state(self):
        return

    def get_dual(self, weight: torch.Tensor) -> float:
        if (not self.regul.provides_fenchel() or not self.loss.provides_fenchel()):
            logger.error("Error: no duality gap available")
            return -float("Inf")
            
        grad2 = None
        grad1, grad2 = self.loss.get_dual_variable(weight, grad2)
        dual = - self.regul.fenchel(grad1, grad2)
        return dual - self.loss.fenchel(grad1)

    def test_stopping_criterion(self, weight: torch.Tensor, iteration: int) -> bool:
        primal = self.loss.eval_tensor(weight) + self.regul.eval_tensor(weight)
        self.best_primal = min(self.best_primal, primal)
        ii = max(int(iteration / self.duality_gap_interval) - 1, 0)
        sec = self.elapsed_time
        optim = self.optim_info[:, ii]
        if (self.best_primal == primal):
            self.best_weight = torch.clone(weight)
        if (self.verbose):
            if (primal == self.best_primal):
                logger.info("Epoch: " + str(iteration) + ", primal objective: " + str(primal) + ", time: " + str(sec))
            else:
                logger.info("Epoch: " + str(iteration) + ", primal objective: " + str(primal) + ", best primal: " + str(self.best_primal) + ", time: " + str(sec))
        optim[0] = iteration
        optim[1] = primal
        optim[5] = sec
        if (self.duality):
            dual = self.get_dual(weight)
            self.best_dual = max(self.best_dual, dual)
            duality_gap = (self.best_primal - self.best_dual) / abs(self.best_primal)
            stop = False            
            if ((iteration / self.duality_gap_interval) >= 4):
                if (self.optim_info[3, int(iteration / self.duality_gap_interval) - 4] == duality_gap):
                    stop = True
                    # TODO Add test to dtype
                    logger.warning("Your problem is prone to numerical instability. It would be safer to use double.")
            if (self.verbose):
                logger.info("Best relative duality gap: " + str(duality_gap))
            optim[2] = self.best_dual
            optim[3] = duality_gap
            if(duality_gap < self.tol):
                stop = True
            elif(duality_gap <= 0 ):
                logger.warning("Your problem is prone to numerical instability. It would be safer to use double.")
                stop = True
            return stop
        else:
            self.previous_weight -= weight
            diff = math.sqrt(torch.square(self.previous_weight) / max(EPSILON, torch.square(self.weight)))
            self.previous_weight = torch.clone(weight)
            optim[4] = diff
            return diff < self.tol

    @abc.abstractmethod
    def solver_init(self, x0 : torch.Tensor):
        return

    @abc.abstractmethod
    def solver_aux(self, x : torch.Tensor):
        return

    @abc.abstractmethod
    def print(self):
        return

    def minibatch(self): 
        return 1