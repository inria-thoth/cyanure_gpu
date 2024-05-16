import torch
from cyanure_pytorch.erm.param.model_param import ModelParameters
from cyanure_pytorch.erm.param.problem_param import ProblemParameters

from cyanure_pytorch.solvers.accelerator import Catalyst, QNing
from cyanure_pytorch.solvers.ista import ISTA_Solver, FISTA_Solver
from cyanure_pytorch.solvers.miso import MISO_Solver
from cyanure_pytorch.solvers.solver import Solver

from cyanure_pytorch.losses.loss import Loss
from cyanure_pytorch.regularizers.regularizer import Regularizer


class Estimator:

    def __init__(self, problem_parameters: ProblemParameters, model_parameters: ModelParameters, optim_info: torch.Tensor):
        """_summary_

        Args:
            problem_parameters (ProblemParameters): _description_
            model_parameters (ModelParameters): _description_
            optim_info (torch.Tensor): _description_
        """
        self.problem_parameters = problem_parameters
        self.model_parameters = model_parameters
        self.optim_info = optim_info

    def is_loss_for_matrices(self, loss: str) -> bool:
        return loss == "SQUARE" or loss == "MULTICLASS-LOGISTIC"

    def is_regression_loss(self, loss: str) -> bool:
        return loss == "SQUARE"

    def is_regul_for_matrices(self, reg: str) -> bool:

        return reg == "L1L2" or reg == "L1LINF"
    
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
            if regul.strong_convexity() > 0:
                solver = MISO_Solver(loss, regul, param)
            else:
                solver = Catalyst(MISO_Solver(loss, regul, param))
        elif solver_type == "CATALYST_MISO":
            solver = Catalyst(param, MISO_Solver(loss, regul, param))
        elif solver_type == "QNING_MISO":
            solver = QNing(param, MISO_Solver(loss, regul, param))
        else:
            solver = None
            raise NotImplementedError("This solver is not implemented !")

        return solver
