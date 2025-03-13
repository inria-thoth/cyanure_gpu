import torch
from cyanureGPU.erm.param.model_param import ModelParameters
from cyanureGPU.erm.param.problem_param import ProblemParameters

from cyanureGPU.solvers.accelerator import Catalyst, QNing
from cyanureGPU.solvers.ista import ISTA_Solver, FISTA_Solver
from cyanureGPU.solvers.solver import Solver

from cyanureGPU.losses.loss import Loss
from cyanureGPU.regularizers.regularizer import Regularizer


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

    def auto_mode(self, loss: Loss, regul: Regularizer):

        n = loss.n()
        if (n < 1000):
            solver_type = "QNING-ISTA"
        else:
            solver_type = "CATALYST-ISTA"

        return solver_type

    def get_solver(self, loss: Loss, regul: Regularizer, param: ModelParameters) -> Solver:
        solver_type = param.solver.upper()

        if "BARZILAI" in solver_type:
            linesearch = True
        else:
            linesearch = False

        solver_type = solver_type.replace('-BARZILAI', '')

        if (solver_type == "AUTO"):
            solver_type = self.auto_mode(loss, regul)
        if solver_type == "ISTA":
            solver = ISTA_Solver(loss, regul, param, linesearch)
        elif solver_type == "QNING-ISTA":
            solver = QNing(param, ISTA_Solver(loss, regul, param, linesearch))
        elif solver_type == "CATALYST-ISTA":
            solver = Catalyst(param, ISTA_Solver(loss, regul, param, linesearch))
        elif solver_type == "FISTA":
            solver = FISTA_Solver(loss, regul, param)
        else:
            solver = None
            raise NotImplementedError("This solver is not implemented !")

        return solver
