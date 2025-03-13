import torch

from cyanureGPU.losses.loss import ProximalPointLoss
from cyanureGPU.erm.param.model_param import ModelParameters

from cyanureGPU.solvers.solver import Solver

from cyanureGPU.constants import DEVICE

from cyanureGPU.erm.algebra import solve_binomial

from cyanureGPU.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class Catalyst(Solver):
    def __init__(self, param: ModelParameters, solver: Solver):
        super().__init__(solver.loss, solver.regul, param)

        self.y = None
        self.dual_var = None
        self.count = None
        self.kappa = None
        self.alpha = None
        self.mu = None
        self.auxiliary_solver = None
        self.loss_ppa = None
        self.freq_restart = param.max_iter + 2 if solver.regul.strong_convexity() > 0 else param.freq_restart
        self.accelerated_solver = True
        self.solver = solver

    def set_dual_variable(self, initial_dual: torch.Tensor) -> None:
        self.dual_var = torch.clone(initial_dual)

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        if (self.verbose):
            self.print()
        self.kappa = self.solver.init_kappa_acceleration(initial_weight)
        self.mu = self.regul.strong_convexity()
        self.count = 0
        self.accelerated_solver = self.kappa > 0  # this->_oldL/(_n) >= _mu;
        if (self.accelerated_solver):
            param2 = ModelParameters()
            param2.max_iter = 1
            param2.duality_gap_interval = 2
            param2.verbose = False
            param2.minibatch = self.minibatch

            self.solver.Li = self.solver.Li + self.kappa
            self.loss_ppa = ProximalPointLoss(self.loss, initial_weight, self.kappa)
            self.auxiliary_solver = type(self.solver)(self.loss_ppa, self.regul, param2, self.solver.linesearch, self.solver.Li)
            if (self.dual_var is not None):
                self.auxiliary_solver.set_dual_variable(self.dual_var)
            self.y = torch.clone(initial_weight)
            self.alpha = 1.0
        else:
            if (self.verbose):
                logger.info("Switching to regular solver, problem is well conditioned")
            self.solver.solver_init(initial_weight)

    def solver_aux(self, weight: torch.Tensor, it: int = -1) -> torch.Tensor:
        if (self.accelerated_solver):
            q = self.mu / (self.mu + self.kappa)
            xold = torch.clone(weight)
            weight, _ = self.auxiliary_solver.solve(self.y, weight)
            alphaold = self.alpha
            self.alpha = solve_binomial(1.0, self.alpha * self.alpha - q, -self.alpha * self.alpha)
            beta = alphaold * (1.0 - alphaold) / (alphaold * alphaold + self.alpha)
            self.count += 1
            if (self.count % self.freq_restart == 0):
                beta = 0
                self.alpha = 1.0
            self.y = torch.clone(xold)
            self.y = -beta * self.y + weight * (1.0 + beta)
            # z = anchor_point
            self.loss_ppa.anchor_point = self.y
        else:
            weight, _ = self.solver.solver_aux(weight, it)

        return weight, None

    def print(self) -> None:
        logger.info("Catalyst Accelerator")


class QNing(Catalyst):

    def __init__(self, param: ModelParameters, solver: Solver):
        super().__init__(param, solver)

        self.l_memory = param.l_memory
        self.initial_h = None
        self.ys = None
        self.ss = None
        self.rhos = None
        self.gk = None
        self.xk = None
        self.Fk = None
        self.etak = None
        self.m = None
        self.skipping_steps = 0
        self.line_search_steps = 0

    def solve(self, initial_weight: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight, fprox = super().solve(initial_weight, weight)
        if (self.verbose):
            logger.info("Total additional line search steps: " + str(self.line_search_steps))
            logger.info("Total skipping l-bfgs steps: " + str(self.skipping_steps))

        return weight, fprox

    def solver_init(self, initial_weight: torch.Tensor) -> None:
        if (len(initial_weight.size()) == 2):
            dim_0_size = initial_weight.size(dim=0) * initial_weight.size(dim=1)
        else:
            dim_0_size = initial_weight.size(dim=0)
        self.ys = torch.empty(dim_0_size, self.l_memory).to(device=DEVICE, non_blocking=True)
        self.ss = torch.empty(dim_0_size, self.l_memory).to(device=DEVICE, non_blocking=True)
        self.rhos = torch.empty(self.l_memory, device=DEVICE)
        super().solver_init(initial_weight)
        if (self.accelerated_solver):
            if (self.verbose):
                logger.info("Memory parameter: " + str(self.l_memory))
            self.h0 = 1.0 / self.kappa
            self.m = 0
            self.etak = 1.0
            self.skipping_steps = 0
            self.line_search_steps = 0

    def solver_aux(self, weight: torch.Tensor, it: int = -1) -> torch.Tensor:
        if (self.accelerated_solver):
            if (self.gk is None):
                weight = self.get_gradient(weight)

            # update variable _y and test
            oldyk = torch.clone(self.y)
            oldxk = torch.clone(weight)
            oldFk = self.Fk
            oldgk = torch.clone(self.gk)
            g = self.get_lbfgs_direction()

            max_iter = 5
            self.auxiliary_solver.save_state()
            for ii in range(max_iter):
                self.y = torch.clone(oldyk)
                self.y.sub_(g, alpha=self.etak)
                self.y.add_(oldgk, alpha=((self.etak - 1.0) / self.kappa))
                weight = self.get_gradient(weight)  # _gk = kappa(x-y)
                if (self.etak == 0 or self.Fk <= (oldFk - (0.25 / self.kappa) * torch.linalg.norm(oldgk)**2)):
                    break
                if (self.Fk > 1.05 * oldFk):
                    self.auxiliary_solver.restore_state()
                    weight = torch.clone(oldxk)
                self.etak /= 2
                self.line_search_steps += 1
                if (ii == max_iter - 1 or self.etak < 0.1):
                    self.etak = 0
            if (self.Fk > 1.05 * oldFk):
                self.auxiliary_solver.restore_state()
                weight = torch.clone(oldxk)
                self.reset_lbfgs()
            else:
                oldyk = self.y - oldyk
                oldgk = self.gk - oldgk
                self.update_lbfgs_matrix(oldyk, oldgk)
            self.etak = max(min(1.0, self.etak * 1.2), 0.1)
        else:
            weight, _ = self.solver.solver_aux(weight, it)

        return weight, None

    def print(self) -> None:
        logger.info("QNing Accelerator")

    def get_lbfgs_direction(self) -> torch.Tensor:
        g = torch.clone(self.gk)
        g_vectorize = torch.flatten(g)
        g_vectorize = self.get_lbfgs_direction_aux(g_vectorize)
        g = torch.reshape(g_vectorize, self.gk.size())

        return g

    def get_lbfgs_direction_aux(self, g: torch.Tensor) -> torch.Tensor:
        # two-loop recursion algorithm
        alphas = torch.Tensor(self.l_memory).to(DEVICE)
        gamma = 1.0 / self.kappa
        for ii in range(self.m - 1, max(self.m - self.l_memory, 0) - 1, -1):
            ind = ii % self.l_memory
            cols = self.ss[:, ind]
            coly = self.ys[:, ind]
            if (ii == self.m - 1):
                gamma = torch.dot(cols, coly) / torch.linalg.norm(coly)**2
            alphas[ind] = self.rhos[ind] * torch.dot(cols, g)
            g.add_(coly, alpha=-alphas[ind])
        g = g * gamma
        for ii in range(max(self.m - self.l_memory, 0), self.m):
            ind = ii % self.l_memory
            cols = self.ss[:, ind]
            coly = self.ys[:, ind]
            beta = self.rhos[ind] * torch.dot(coly, g)
            g.add_(cols, alpha=alphas[ind] - beta)
        return g

    def update_lbfgs_matrix(self, sk: torch.Tensor, yk: torch.Tensor) -> None:

        sk_vectorize = torch.flatten(sk)
        yk_vectorize = torch.flatten(yk)

        theta = torch.dot(sk_vectorize, yk_vectorize)
        if (theta > 1e-12):
            ind = self.m % self.l_memory
            self.ys[:, ind] = yk_vectorize
            self.ss[:, ind] = sk_vectorize
            self.rhos[ind] = 1.0 / theta
            self.m += 1
        else:
            self.skipping_steps += 1

    def reset_lbfgs(self) -> None:
        self.m = 0

    def get_gradient(self, weight: torch.Tensor) -> torch.Tensor:
        self.loss_ppa.anchor_point = self.y
        weight, fprox = self.auxiliary_solver.solve(self.y, weight)
        self.gk = torch.clone(self.y)
        self.gk = torch.add(self.gk * self.kappa, weight, alpha=-self.kappa)
        self.Fk = self.loss_ppa.eval_tensor(weight, None, None, fprox) + self.regul.eval_tensor(weight)

        return weight
