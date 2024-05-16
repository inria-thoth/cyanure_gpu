import copy

import torch
import torch.nn

from cyanure_pytorch.erm.erm import Estimator
from cyanure_pytorch.erm.param.model_param import ModelParameters
from cyanure_pytorch.erm.param.problem_param import ProblemParameters
from cyanure_pytorch.logger import setup_custom_logger
from cyanure_pytorch.losses.logistic import LogisticLoss
from cyanure_pytorch.losses.loss import Loss
from cyanure_pytorch.losses.square import SquareLoss
from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.regularizers.ridge import Ridge
from cyanure_pytorch.regularizers.lasso import Lasso
from cyanure_pytorch.solvers.accelerator import Catalyst, QNing
from cyanure_pytorch.solvers.ista import ISTA_Solver, FISTA_Solver
from cyanure_pytorch.solvers.miso import MISO_Solver
from cyanure_pytorch.solvers.solver import Solver
from cyanure_pytorch.constants import EPSILON

from typing import Tuple

logger = setup_custom_logger("INFO")

class SimpleErm(Estimator):

    global EPSILON

    def __init__(self, initial_weight: torch.Tensor, weight: torch.Tensor, problem_parameters: ProblemParameters, model_parameters: ModelParameters, optim_info: torch.Tensor, dual_variable: torch.Tensor):
        super().__init__(problem_parameters, model_parameters, optim_info)
        self.initial_weight = initial_weight
        self.weight = weight
        self.dual_variable = dual_variable

    def solve_problem(self, features: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if (self.model_parameters.verbose):
            logger.info("Matrix X, n=" + str(features.size(dim=1)) + ", p=" + str(features.size(dim=0)))
            
        self.verify_input(features)
        loss = self.get_loss(features, labels)
        regul = self.get_regularization()
        solver = None

        if (self.model_parameters.max_iter == 0):
            parameter_tmp = self.model_parameters
            parameter_tmp.verbose = False
            solver = ISTA_Solver(loss,  regul, parameter_tmp, False)
            solver.eval(self.initial_weight)
            self.weight = torch.clone(self.initial_weight)
        else:

            if (solver is None):
                regul.strong_convexity()
                solver = self.get_solver(loss, regul, self.model_parameters)

            if (solver is None):
                self.weight = torch.clone(self.initial_weight)
                return None

            if (self.problem_parameters.intercept):
                new_initial_weight = loss.set_intercept(self.initial_weight)
            else:
                new_initial_weight = torch.clone(self.initial_weight)
            
            if (self.dual_variable is not None and self.dual_variable.size(dim=0) != 0):
                solver.set_dual_variable(self.dual_variable)
            
            self.weight, fprox = solver.solve(new_initial_weight, self.weight)

            if (self.problem_parameters.intercept):
                self.weight = loss.reverse_intercept(self.weight)

        if (self.problem_parameters.regul == "L1"):
            self.weight[abs(self.weight) < EPSILON] = 0

        if (self.weight is None):
            self.weight = self.initial_weight

        return solver.get_optim_info(), self.weight

    def verify_input(self, X: torch.Tensor) -> None:
        if (self.problem_parameters.intercept):
            if (X.size(dim=0) + 1 != self.initial_weight.size(dim=0)):
                logger.error("Dimension of initial point is not consistent. With intercept, if X is m x n, w0 should be (n+1)-dimensional.")
                return None
        else:
            if (X.size(dim=0) != self.initial_weight.size(dim=0)):
                logger.error("Dimension of initial point is not consistent. If X is m x n, w0 should be n-dimensional.")
                return None

        if (self.model_parameters.max_iter < 0):
            raise ValueError("Maximum number of iteration must be positive")
        if (self.problem_parameters.lambda_1 < 0):
            raise ValueError("Penalty term must be positive")
        if (self.model_parameters.tol < 0):
            raise ValueError("Tolerance for stopping criteria must be positive")


    def get_regularization(self) -> Regularizer:
        regul = None

        self.problem_parameters.regul = self.problem_parameters.regul.upper()
        if self.problem_parameters.regul == "L2":
            regul = Ridge(self.problem_parameters)
        elif self.problem_parameters.regul == "L1":
            regul = Lasso(self.problem_parameters)
        elif self.problem_parameters.regul is None:
            pass
        else:
            logger.error("Not implemented, no regularization is chosen")
        
        return regul
    

    def get_solver(self, loss: Loss, regul: Regularizer, param: ModelParameters) -> Solver:
        solver_type = param.solver.upper()

        if "BARZILAI" in solver_type:
            linesearch = True
        else:
            linesearch = False

        solver_type = solver_type.replace('_BARZILAI', '')

        if (solver_type == "AUTO"):
            L = loss.lipschitz()
            n = loss.n()
            lambda_1 = regul.strong_convexity()
            if (n < 1000):
                solver_type = "QNING_ISTA"
            elif (lambda_1 < L / (100 * n)):
                solver_type = "QNING_MISO"
            else:
                solver_type = "CATALYST_MISO"
        if solver_type == "ISTA":
            solver = ISTA_Solver(loss, regul, param, linesearch)
        elif solver_type == "QNING_ISTA":
            solver = QNing(param, ISTA_Solver(loss, regul, param, linesearch))
        elif solver_type == "CATALYST_ISTA":
            solver = Catalyst(param, ISTA_Solver(loss, regul, param, linesearch))
        elif solver_type == "FISTA":
            solver = FISTA_Solver(loss, regul, param)
        elif solver_type == "MISO":
            solver = MISO_Solver(loss, regul, param) if regul.strong_convexity() > 0 else Catalyst(MISO_Solver(loss, regul, param))
        elif solver_type == "CATALYST_MISO":
            solver = Catalyst(param, MISO_Solver(loss, regul, param))
        elif solver_type == "QNING_MISO":
            solver = QNing(param, MISO_Solver(loss, regul, param))
        else:
            solver = None
            raise NotImplementedError("This solver is not implemented !")
            
        return solver


    def get_loss(self, data: torch.Tensor, y: torch.Tensor):
        loss = None

        self.problem_parameters.loss = self.problem_parameters.loss.upper()
        if (self.problem_parameters.loss == "SQUARE"):
            loss = SquareLoss(data, y, self.problem_parameters.intercept)
        elif (self.problem_parameters.loss == "LOGISTIC"):
            loss = LogisticLoss(data, y, self.problem_parameters.intercept)
        else:
            logger.error("Not implemented, square loss is chosen by default")
            loss = SquareLoss(data, y, self.problem_parameters.intercept)
        
        return loss