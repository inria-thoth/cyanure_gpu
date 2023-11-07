import torch
from cyanure_pytorch.erm.param.model_param import ModelParameters
from cyanure_pytorch.erm.param.problem_param import ProblemParameters

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
        return loss == "SQUARE" or loss == "MULTI_LOGISTIC"
    

    def is_regression_loss(self, loss: str) -> bool:
        return loss == "SQUARE"
    

    def is_regul_for_matrices(self, loss: str) -> bool:
    
        return reg == "L1L2" or reg == "L1LINF"


    
