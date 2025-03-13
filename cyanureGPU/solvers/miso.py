import torch
from random import random

from cyanure_pytorch.losses.loss import Loss
from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.model_param import ModelParameters

from cyanure_pytorch.logger import setup_custom_logger
from cyanure_pytorch.solvers.incremental import IncrementalSolver

from cyanure_pytorch.constants import DEVICE

logger = setup_custom_logger("INFO")

class MISO_Solver(IncrementalSolver):

    def __init__(self, loss : Loss, regul : Regularizer, param: ModelParameters, Li : torch.Tensor = None):
        super().__init__(loss, regul, param)

        self.minibatch = 1
        self.mu = self.regul.strong_convexity() if self.regul.id == "L2" or self.regul.id == "ELASTICNET" else 0
        self.kappa = self.loss.kappa()
        if (self.loss.id == "PPA"):
            self.mu += self.kappa
        
        self.isprox = (self.regul.id != "L2" or self.regul.intercept) and self.regul.id != "None"
        self.is_lazy = self.isprox and self.regul.is_lazy() and self.loss.is_sparse()
        self.extern_zis = False

        self.zis = None
        self.barz = None
        self.oldy = None
        self.zis2 = None
        self.barz2 = None
        self.oldy2 = None
        self.delta = 0.0
        self.count = 0
        self.count2 = 0
        
    def set_dual_variable(self, dual0 : torch.Tensor) -> None:
        self.extern_zis = True
        self.zis.copy_(dual0)

    def solve(self, y : torch.Tensor, x : torch.Tensor) -> torch.Tensor:
        if (self.count > 0 and (self.count % 10) != 0):
            ref_barz = self.barz if self.isprox else x
            ref_barz.add_(self.oldy, alpha=-self.kappa / self.mu) # necessary to have PPA loss here
            ref_barz.add_(y, alpha=self.kappa / self.mu)
            is_lazy = self.isprox and self.regul.is_lazy() and self.loss.is_sparse()
            if (self.isprox and not is_lazy):
                self.regul.prox(ref_barz, x, 1.0 / self.mu)
        elif (self.count == 0):
            x = y.clone() #just to have the right size
        if (self.loss.id == "PPA"):
            self.oldy = self.loss.anchor_point.clone()
        weight = super().solve(x, x)

        return weight

    def save_state(self) -> None:
        self.count2 = self.count
        self.barz2 = self.barz.clone()
        self.zis2 = self.zis.clone()
        self.oldy2 = self.oldy.clone()
    
    def restore_state(self) -> None:
        self.count = self.count2
        self.barz = self.barz2.clone()
        self.zis = self.zis2.clone()
        self.oldy = self.oldy2.clone()

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        # initial point will be in fact _z of PPA
        if (self.count == 0):
            super().solver_init(initial_weight)
            self.delta = min(1.0, self.n * self.mu / (2 * self.L))
            if (self.non_uniform_sampling):
                beta = 0.5 * self.mu * self.n
                if (torch.max(self.Li) <= beta):
                    self.non_uniform_sampling = False
                    self.delta = 1.0
                elif (torch.min(self.Li) >= beta):
                    # no change
                    pass
                else:
                    self.qi = self.Li.clone()
                    self.qi[self.qi < beta] = beta
                    self.qi *= 1.0 / torch.sum(self.qi)
                    tmp = self.qi.clone()
                    tmp = 1.0 / tmp
                    tmp = torch.pow(tmp, 2) + self.Li
                    self.L = torch.max(tmp) / self.n
                    self.init_nonu_sampling()
                    self.delta = min(self.n * torch.min(self.qi), self.n * self.mu / (2 * self.L))
            if (self.non_uniform_sampling):
                self.delta = min(self.delta, self.n * torch.min(self.qi))
            if (self.isprox):
                self.barz = initial_weight.clone() # if PPA, x0 should be the anchor point and _barz = X*_Zis + x0
            
            if (len(initial_weight.size()) > 1):
                self.init_dual_variables_2d(initial_weight)
            else:
                self.init_dual_variables_1d(initial_weight)
    
    def solver_aux(self, weight : torch.Tensor) -> torch.Tensor:
        ref_barz = self.barz if self.isprox else weight
        if (self.count % 10 == 0):
            if (self.loss.id == "PPA"):
                ref_barz = self.loss.anchor_point.clone()
                ref_barz *= self.kappa / self.mu
            else:
                ref_barz.fill_(0)
            if (self.count > 1 or self.extern_zis):
                ref_barz = self.loss.add_feature(self.zis, ref_barz, 1.0 / (self.n * self.mu))
            if (self.isprox and not self.is_lazy):
                weight = self.regul.prox(ref_barz, 1.0 / self.mu)
        self.count += 1
        for ii in range(self.n):
            ind = self.nonu_sampling() if self.non_uniform_sampling else int(random() % self.n)
            scal = 1.0 / (self.qi[ind] * self.n) if self.non_uniform_sampling else 1.0
            deltas = scal * self.delta
            if (self.is_lazy):
                # TODO Only for sparse problem
                # TODO Remove
                indices = self.loss.get_coordinates(ind)
                weight = self.regul.lazy_prox(ref_barz, indices, 1.0 / self.mu)

            if (len(weight.size()) > 1):
                ref_barz, weight = self.solver_aux_aux_2d(weight, ref_barz, ind, deltas)
            else:
                ref_barz, weight = self.solver_aux_aux_1d(weight, ref_barz, ind, deltas)

            if (self.isprox and (not self.is_lazy or ii == self.n - 1)):
                weight = self.regul.prox(ref_barz, 1.0 / self.mu)
        
        return weight
    
    def print(self) -> None:
        logger.info("MISO Solver")
        super().print()

    def init_dual_variables_1d(self, initial_weight: torch.Tensor) -> None:
        if (self.zis is None or self.zis.size(dim=0) != self.n):
            self.zis = torch.zeros(self.n, device=DEVICE)
    
    def init_dual_variables_2d(self, initial_weight: torch.Tensor) -> None:
        nclasses =  initial_weight.size(dim=0) if self.loss.transpose() else initial_weight.size(dim=1)
        if (self.zis is None or self.zis.size(dim=1) != self.n or self.zis.size(dim=0) != nclasses):
            self.zis = torch.zeros((nclasses, self.n), device=DEVICE)
    
    def solver_aux_aux_1d(self, weight : torch.Tensor, ref_barz : torch.Tensor, ind : int, deltas : float):
        oldzi = self.zis[ind]
        self.zis[ind] = (1.0 - deltas) * self.zis[ind] + deltas * (-self.loss.scal_grad(weight, ind))
        ref_barz = self.loss.add_feature(ind, (self.zis[ind] - oldzi) / (self.n * self.mu), None)

        return ref_barz, weight

    def solver_aux_aux_2d(self, weight : torch.Tensor, ref_barz : torch.Tensor, ind : int, deltas : float):
        oldzi = self.zis[:, ind]
        newzi = self.zis[:, ind]
        newzi = self.loss.scal_grad(weight, ind)
        newzi = oldzi * (1.0 - deltas) + newzi * (-deltas)
        oldzi.sub_(newzi)
        oldzi *= (-1.0 / (self.n * self.mu))
        ref_barz = self.loss.add_feature(ind, oldzi, None)

        return ref_barz, weight
