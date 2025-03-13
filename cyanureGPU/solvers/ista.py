import torch

from cyanureGPU.losses.loss import Loss
from cyanureGPU.regularizers.regularizer import Regularizer
from cyanureGPU.erm.param.model_param import ModelParameters

from cyanureGPU.solvers.solver import Solver

from cyanureGPU.constants import EPSILON, DEVICE

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class ISTA_Solver(Solver):

    global EPSILON

    def __init__(self, loss: Loss, regul: Regularizer, param: ModelParameters, linesearch: bool, Li: torch.Tensor = None):
        super().__init__(loss, regul, param)
        self.L = 0
        if Li is not None:
            self.Li = torch.clone(Li)
            self.L = torch.max(self.Li) / 100.0

        self.linesearch = linesearch
        if linesearch:
            self.sbb = None
            self.xbb = None

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        if (self.verbose):
            self.print()
        if (self.L == 0):
            self.Li = self.loss.lipschitz_li(self.Li)
            self.L = torch.max(self.Li) / 100.0

    def solver_aux(self, weight: torch.Tensor, it: int) -> torch.Tensor:

        iter = 1
        if hasattr(self.loss, 'loss'):
            precompute = self.loss.loss.pre_compute(weight)
        else:
            precompute = self.loss.pre_compute(weight)
        fx = self.loss.eval_tensor(weight, None, precompute)
        grad = self.loss.grad(weight, None, precompute)

        alpha_max = 10e30/self.L
        alpha_min = 10e-30/self.L

        if (self.linesearch and it > 1):
            self.sbb.sub_(grad)
            self.xbb.sub_(weight)
            alpha = torch.dot(self.sbb.view(-1), self.sbb.view(-1)) / torch.norm(self.sbb)**2
            alpha = torch.min(torch.max(alpha, alpha_min), alpha_max)
            self.L = 1 / alpha

        while (iter < self.max_iter_backtracking):
            tmp2 = weight.clone()
            # Perform in-place subtraction for better performance
            tmp2 = tmp2 - grad / self.L
            tmp = self.regul.prox(tmp2, 1.0 / self.L)
            fprox = self.loss.eval_tensor(tmp)
            tmp2 = tmp.clone()
            tmp2 = tmp2 - weight
            dot_product = torch.tensordot(grad, tmp2, dims=len(grad.shape))
            norm_squared = torch.pow(torch.norm(tmp2), 2)
            if ((self.linesearch and it > 1) or fprox <= fx + dot_product + 0.5 * self.L * norm_squared + EPSILON):
                break
            self.L *= 1.5
            if (self.verbose):
                logger.info("new value for L: " + str(self.L))
            iter += 1
            if (iter == self.max_iter_backtracking):
                logger.warning("Warning: maximum number of backtracking iterations has been reached")

        if (self.linesearch):
            self.sbb = grad.clone()
            self.xbb = weight.clone()

        weight = torch.clone(tmp)

        return weight, fprox

    def print(self) -> None:
        logger.info("ISTA Solver")

    def init_kappa_acceleration(self, initial_weight: torch.Tensor) -> float:
        self.solver_init(initial_weight)
        return self.L


class FISTA_Solver(ISTA_Solver):
    def __init__(self, loss: Loss, regul: Regularizer, param: ModelParameters):
        super().__init__(loss, regul, param, linesearch=False)

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        super().solver_init(initial_weight)
        self.t = torch.Tensor([1.0]).to(DEVICE)
        self.y = torch.clone(initial_weight)

    def solver_aux(self, weight: torch.Tensor, it: int = -1) -> torch.Tensor:
        # Step 1: Clone the weight tensor to avoid mutating it unintentionally.
        diff = torch.clone(weight)

        # Step 2: Call the parent class's solver_aux method and update `weight`.
        weight, _ = super().solver_aux(self.y, it)
        self.y = weight  # Update y to the new weight from super() call.

        # Step 3: Calculate the difference more robustly, adding a small epsilon.
        diff = diff - weight
        eps = torch.finfo(diff.dtype).eps  # Small epsilon to avoid very small diffs being zeroed out
        diff = torch.where(torch.abs(diff) < eps, torch.tensor(0.0, dtype=diff.dtype), diff)

        # Step 4: Update `self.t` with a stable expression.
        old_t = self.t.clone().detach()
        self.t = (1.0 + torch.sqrt(1.0 + 4.0 * torch.pow(old_t, 2))) / 2.0

        # Step 5: Perform a numerically stable update to `self.y`
        update_factor = (1.0 - old_t) / self.t
        self.y = self.y + diff * update_factor

        return weight, _

    def print(self) -> None:
        logger.info("FISTA Solver")
