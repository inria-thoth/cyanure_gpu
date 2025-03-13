import torch

from cyanureGPU.constants import DEVICE


class ProblemParameters:
    lambda_1: float
    lambda_2: float
    lambda_3: float
    intercept: bool
    regul: str
    loss: str

    def __init__(self, lambda_1: float = 0, lambda_2: float = 0, lambda_3: float = 0, intercept: bool = False,
                 regul: str = "NONE", loss: str = "SQUARE"):
        self.lambda_1 = torch.tensor([lambda_1], device=DEVICE)
        self.lambda_2 = torch.tensor([lambda_2], device=DEVICE)
        self.lambda_3 = torch.tensor([lambda_3], device=DEVICE)
        self.intercept = intercept
        self.regul = regul
        self.loss = loss
