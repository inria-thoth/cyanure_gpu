import torch

from cyanure.losses.loss import Loss
from cyanure.regularizers.regularizer import Regularizer
from cyanure.erm.param.model_param import ModelParameters

from cyanure.solvers.solver import Solver

from cyanure.constants import EPSILON

from cyanure.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class ISTA_Solver(Solver):

        global EPSILON
    
        def __init__(self, loss : Loss, regul : Regularizer, param: ModelParameters, Li : torch.Tensor = None):
            super().__init__(loss, regul, param)

            self.L = 0
            if (Li):
                self.Li = torch.clone(Li)
                self.L = torch.max(self.Li) / 100

        def solver_init(self, initial_weight: torch.Tensor) -> None:
            if (self.L == 0):
                self.Li = self.loss.lipschitz_li(self.Li)
                self.L = torch.max(self.Li) / 100
        
        def solver_aux(self, weight : torch.Tensor) -> torch.Tensor:
            iter = 1
            fx = self.loss.eval_tensor(weight)
            grad = self.loss.grad(weight)
            while (iter < self.max_iter_backtracking):
                tmp2 = torch.clone(weight)
                tmp2 = (-1.0 / self.L) * grad + tmp2
                tmp = self.regul.prox(tmp2, 1.0 / self.L)
                fprox = self.loss.eval_tensor(tmp)
                tmp2 = torch.clone(tmp)
                tmp2 = torch.sub(tmp2, weight)

                if (fprox <= fx + torch.dot(grad, tmp2) + 0.5 * self.L * torch.pow(torch.linalg.vector_norm(tmp2), 2) + EPSILON):
                    weight = torch.clone(tmp)
                    break
                self.L *= 1.5
                if (self.verbose):
                    logger.info("new value for L: " + str(self.L))
                iter += 1
                if (iter == self.max_iter_backtracking):
                    logger.warning("Warning: maximum number of backtracking iterations has been reached")
            
            return weight
        
        def print(self) -> None:
            logger.info("ISTA Solver")
        
        def init_kappa_acceleration(self, initial_weight : torch.Tensor) -> float:
            self.solver_init(initial_weight)
            return self.L
        

"""
    template <typename loss_type>
    class FISTA_Solver final : public ISTA_Solver<loss_type>
    {
    public:
        USING_SOLVER
        FISTA_Solver(const loss_type& loss, const Regularizer<D, PointerType>& regul, const ParamSolver<FeatureType>& param) : ISTA_Solver<loss_type>(loss, regul, param) {}

    protected:
        virtual void solver_init(const D& x0)
        {
            ISTA_Solver<loss_type>::solver_init(x0)
            _t = FeatureType(1.0)
            _y.copy(x0)
        }
        
        virtual void solver_aux(D& x)
        {
            ISTA_Solver<loss_type>::solver_aux(_y)
            D diff
            diff.copy(x)
            x.copy(_y)
            diff.sub(x)
            const FeatureType old_t = _t
            _t = (1.0 + sqrt(1 + 4 * _t * _t)) / 2
            _y.add(diff, (FeatureType(1.0) - old_t) / _t)
        }
        
        virtual void print() const
        {
            logging(logINFO) << "FISTA Solver"
        }

        FeatureType _t
        D _y
    }
"""