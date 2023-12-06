class ProblemParameters:
    lambda_1: float
    lambda_2: float
    lambda_3: float
    intercept: bool
    regul : str
    loss : str

    def __init__(self, lambda_1: float = 0, lambda_2: float = 0, lambda_3: float = 0, intercept : bool = False, regul = None, loss="SQUARE"):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.intercept = intercept
        self.regul = regul
        self.loss = loss