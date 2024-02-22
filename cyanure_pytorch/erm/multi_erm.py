import torch

from cyanure_pytorch.erm.erm import Estimator
from cyanure_pytorch.erm.simple_erm import SimpleErm
from cyanure_pytorch.erm.param.problem_param import ProblemParameters
from cyanure_pytorch.erm.param.model_param import ModelParameters
from cyanure_pytorch.logger import setup_custom_logger
from cyanure_pytorch.losses.logistic import LogisticLoss
from cyanure_pytorch.losses.loss import Loss
from cyanure_pytorch.losses.square_multiclass import SquareLossMat
from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.regularizers.ridge import Ridge
from cyanure_pytorch.solvers.accelerator import Catalyst, QNing
from cyanure_pytorch.solvers.ista import ISTA_Solver
from cyanure_pytorch.solvers.solver import Solver
from cyanure_pytorch.losses.multi_class_logistic import MultiClassLogisticLoss
from cyanure_pytorch.losses.loss_matrix import LossMat
from cyanure_pytorch.regularizers.regularizer_matrix import RegMat, RegVecToMat

from cyanure_pytorch.constants import EPSILON, NUMBER_OPTIM_PROCESS_INFO, DEVICE

from typing import Tuple

logger = setup_custom_logger("INFO")

from typing import Tuple
import time

class MultiErm(Estimator):

    global EPSILON

    def __init__(self, initial_weight : torch.Tensor, weight : torch.Tensor, problem_parameters: ProblemParameters, model_parameters: ModelParameters, optim_info: torch.Tensor, dual_variable : torch.Tensor):
        super().__init__(problem_parameters, model_parameters, optim_info)
        self.initial_weight = initial_weight
        self.weight = weight
        self.dual_variable = dual_variable

    # X is p x n
    # y is nclasses x n
    # W0 is p x nclasses if no intercept (or p+1 x nclasses with intercept)
    # prediction model is   W0^FeatureType X  gives  nclasses x n
    def solve_problem_vector(self, features : torch.Tensor, labels_vector : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import time 
        
        self.verify_input(features)
        # print(f"Starting Memory Usage Vector: {torch.cuda.memory_allocated() / 1e6} MB")
        nclass = torch.max(labels_vector) + 1
        loss_string = self.problem_parameters.loss.upper()

        if (super().is_regression_loss(loss_string) or not super().is_loss_for_matrices(loss_string)):
            n = labels_vector.size(dim=0)
            
            import numpy as np
            initial_time = time.time()
            labels_np = np.full((int(nclass.item()), n), fill_value=-1.0)

            # Assuming labels_vector is a PyTorch tensor
            labels_vector_np = labels_vector.cpu().numpy()

            # Set the corresponding elements to 1.0
            labels_np[labels_vector_np.astype("int32"), np.arange(n)] = 1.0

            # Convert NumPy array to PyTorch tensor
            labels = torch.tensor(labels_np.astype("float32"), device=DEVICE)
            print(f"Initialize ERM : {time.time() - initial_time}")
            
            erm_tmp = MultiErm(self.initial_weight, self.weight, self.problem_parameters, self.model_parameters, self.optim_info, dual_variable=self.dual_variable)
            return erm_tmp.solve_problem_matrix(features, labels)

        if (self.model_parameters.verbose):
            logger.info("Matrix X, n=" + str(features.size(dim=1)) + ", p=" + str(features.size(dim=0)))
            pass

        loss = MultiClassLogisticLoss(features, labels_vector, self.problem_parameters.intercept)
        if (loss_string != 'MULTICLASS-LOGISTIC'):
            logger.error("Multilog loss is the only multi class implemented loss!")
            logger.info("Multilog loss is used!")
        
        transpose = loss.transpose()

        regul = self.get_regul_mat(nclass, transpose)

        return self.solve_mat(loss, regul)


    # X is p x n
    # y is nclasses x n
    # W0 is p x nclasses if no intercept (or p+1 x nclasses with intercept)
    # prediction model is   W0^FeatureType X  gives  nclasses x n
    def solve_problem_matrix(self, features : torch.Tensor, labels : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.verify_input(features)
        # print(f"Starting Memory Usage Matrix: {torch.cuda.memory_allocated() / 1e6} MB")
        loss_string = self.problem_parameters.loss.upper()
        regul_string = self.problem_parameters.regul.upper()

        if (super().is_regul_for_matrices(regul_string) or super().is_loss_for_matrices(loss_string)):
            if (self.model_parameters.verbose):
                logger.info("Matrix X, n=" + str(features.size(dim=1)) + ", p=" + str(features.size(dim=0)))
                pass

            loss = self.get_loss_matrix(features, labels)

            n_class = self.initial_weight.size(dim=1)

            transpose = loss.transpose()

            regul = self.get_regul_mat(n_class, transpose)
            # print(f"Starting Memory Usage Before Solve: {torch.cuda.memory_allocated() / 1e6} MB")
            return self.solve_mat(loss, regul)
        else:
            self.weight = self.initial_weight
            n_class = self.initial_weight.size(dim=1)
            duality_gap_interval = max(self.model_parameters.duality_gap_interval, 1)
            self.optim_info = torch.zeros([n_class, NUMBER_OPTIM_PROCESS_INFO, int(max(self.model_parameters.max_iter / duality_gap_interval, 1))])
            parameter_tmp = self.model_parameters
            parameter_tmp.verbose = False
            if (self.model_parameters.verbose):
                logger.info("Matrix X, n=" + str(features.size(dim=1)) + ", p=" + str(features.size(dim=0)))
                pass
            initial_time = time.time()
            for ii in range (n_class):
                optim_info_col = torch.zeros([1, NUMBER_OPTIM_PROCESS_INFO, int(max(self.model_parameters.max_iter / duality_gap_interval, 1))])
                initial_weight_col = self.initial_weight[:, ii]
                weight_col = self.weight[:, ii]
                labels_col = labels[ii, :]
                if (self.dual_variable.size(dim=0) == n_class):
                    dualcol = self.dual_variable[ii]
                problem_configuration = SimpleErm(initial_weight_col, weight_col, parameter_tmp, self.model_parameters, optim_info_col, dualcol)
                optim_info_col, weight_col = problem_configuration.solve_problem(features, labels_col)
                if (self.dual_variable.size(dim=0) == n_class):
                    self.dual_variable[ii] = problem_configuration.dual_variable
            
                self.weight[:, ii] = weight_col
                self.optim_info[ii] = optim_info_col
                if (self.model_parameters.verbose):
                    noptim = optim_info_col.size(dim=2) - 1;
                    logger.info("Solver " + ii + " has terminated after " + optim_info_col(0, 0, noptim) + " epochs in " + optim_info_col(0, 5, noptim) + " seconds")
                    if (optim_info_col[0, 4, noptim] == 0):
                        logger.info("   Primal objective: " + optim_info_col(0, 1, noptim) + ", relative duality gap: " + optim_info_col(0, 3, noptim))
                    else:
                        logger.info("   Primal objective: " + optim_info_col(0, 1, noptim) + ", tol: " + optim_info_col(0, 4, noptim))
                

            final_time = time.time()
            if (self.model_parameters.verbose):
                logger.info("Time for the one-vs-all strategy")
                logger.info("Elapsed time: " + str(final_time - initial_time))

        return self.optim_info, self.weight

    def verify_input(self, X : torch.Tensor) -> None:
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

    def get_loss_matrix(self, data : torch.Tensor, y : torch.Tensor) -> Loss:
        loss = None

        self.problem_parameters.loss = self.problem_parameters.loss.upper()
        if (self.problem_parameters.loss == "SQUARE"):
            loss = SquareLossMat(data, y, self.problem_parameters.intercept)
        elif (self.problem_parameters.loss == "LOGISTIC"):
            loss = LossMat(LogisticLoss(data, y, self.problem_parameters.intercept))
        elif (self.problem_parameters.loss == "SQHINGE"):
            loss = LossMat(SquaredHingeLoss(data, y, self.problem_parameters.intercept))
        elif (self.problem_parameters.loss == "SAFE_LOGISTIC"):
            loss = LossMat(SafeLogisticLoss(data, y, self.problem_parameters.intercept))
        else:
            logger.error("Not implemented, square loss is chosen by default")
            loss = SquareLossMat(data, y, self.problem_parameters.intercept)
        return loss

    def solve_mat(self, loss : Loss, regul : Regularizer) -> Tuple[torch.Tensor, torch.Tensor]:

        import time 
        initial_time = time.time()
        solver = None
        if (self.model_parameters.max_iter == 0):
            parameter_tmp = self.model_parameters
            parameter_tmp.verbose = False
            solver = ISTA_Solver(loss, regul, parameter_tmp)
            if (loss.transpose()):
                initial_weight_transposed = torch.transpose(self.initial_weight, 0, 1)
                solver.eval(self.initial_weight_transposed)
            else:
                solver.eval(self.initial_weight)
            self.weight = torch.clone(self.initial_weight)        
        else:
            solver = self.get_solver(loss, regul, self.model_parameters)
            
            if (solver is None):
                self.weight = torch.clone(self.initial_weight)
                return solver.get_optim_info(), self.weight
            # print(f"Starting Memory Usage Before Inner Solve: {torch.cuda.memory_allocated() / 1e6} MB")
            new_initial_weight = None
            if (self.problem_parameters.intercept):
                loss.set_intercept(self.initial_weight, new_initial_weight);
            else:
                new_initial_weight = self.initial_weight
            if (self.dual_variable is not None and self.dual_variable.size(dim=0) != 0):
                solver.set_dual_variable(self.dual_variable)
            # print(f"Intermediate Memory Usage Before Inner Solve: {torch.cuda.memory_allocated() / 1e6} MB")
            if (loss.transpose()):
                weight_transposed = None
                initial_weight_transposed = torch.transpose(new_initial_weight, 0, 1)
                print(f"Solve Mat: {time.time() - initial_time}")
                weight_transposed = solver.solve(initial_weight_transposed, weight_transposed)
                self.weight = torch.transpose(weight_transposed, 0, 1)
            else:
                print(f"Solve Mat: {time.time() - initial_time}")
                self.weight = solver.solve(new_initial_weight, self.weight)

            if (self.problem_parameters.intercept):
                loss.reverse_intercept(self.weight);

        if (self.problem_parameters.regul == "L1"):
            self.weight[abs(self.weight) < EPSILON] = 0

        return solver.get_optim_info(), self.weight

    def get_solver(self, loss : Loss, regul : Regularizer, param : ModelParameters) -> Solver:
        solver_type = param.solver.upper()

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
            solver = ISTA_Solver(loss, regul, param)
        elif solver_type == "QNING_ISTA":
            solver = QNing(param, ISTA_Solver(loss, regul, param))
        elif solver_type == "CATALYST_ISTA":
            solver = Catalyst(param, ISTA_Solver(loss, regul, param))
        elif solver_type == "FISTA":
            solver = FISTA_Solver(loss, regul, param)
        elif solver_type == "SVRG":
            solver = SVRG_Solver(loss, regul, param)
        elif solver_type == "MISO":
            solver = MISO_Solver(loss, regul, param) if regul.strong_convexity() > 0 else Catalyst(MISO_Solver(loss, regul, param))
        elif solver_type == "SVRG_UNIFORM":
            new_model_param = param
            new_model_param.non_uniform_sampling = False
            solver = SVRG_Solver(loss, regul, new_model_param)
        elif solver_type == "CATALYST_SVRG":
            solver = Catalyst(param, SVRG_Solver(loss, regul, param))
        elif solver_type == "QNING_SVRG":
            solver = QNing(param, SVRG_Solver(loss, regul, param))
        elif solver_type == "CATALYST_MISO":
            solver = Catalyst(param, MISO_Solver(loss, regul, param))
        elif solver_type == "QNING_MISO":
            solver = QNing(param, MISO_Solver(loss, regul, param))
        elif solver_type == "ACC_SVRG":
            solver = Acc_SVRG_Solver(loss, regul, param)
        else:
            solver = None
            raise NotImplementedError("This solver is not implemented !")
            
        return solver

    def get_regul_mat(self, num_class : int, transpose : bool) -> Regularizer:
        regul = None

        self.problem_parameters.regul = self.problem_parameters.regul.upper()
        regularizer_string = self.problem_parameters.regul
        if (regularizer_string == "L2"):
            regul = RegVecToMat(Ridge(self.problem_parameters), self.problem_parameters) if transpose else RegMat(Ridge(self.problem_parameters), self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "L1"):
            regul = RegVecToMat(Lasso(self.problem_parameters), self.problem_parameters) if transpose else RegMat(Lasso(self.problem_parameters), self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "ELASTICNET"):
            regul = RegVecToMat(ElasticNet(self.problem_parameters), self.problem_parameters) if transpose else RegMat(ElasticNet(self.problem_parameters), self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "L1BALL"):
            regul = RegMat(L1Ball(self.problem_parameters), self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "L2BALL"):
            regul = RegMat(L2Ball(self.problem_parameters), self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "L1L2"):
            regul = MixedL1L2(self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "L1L2_L1"):
            regul = MixedL1L2_L1(self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "L1LINF"):
            regul = MixedL1Linf(self.problem_parameters, num_class, transpose)
        elif (regularizer_string == "FUSEDLASSO"):
            regul = RegMat(FusedLasso(self.problem_parameters), self.problem_parameters, num_class, transpose)
        elif (regularizer_string is None):
            pass
        else:
            logger.error("Not implemented, no regularization is chosen")

        return regul