from typing import Tuple
import torch

from cyanure_pytorch.regularizers.regularizer import Regularizer
from cyanure_pytorch.erm.param.problem_param import ProblemParameters

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class RegMat(Regularizer):

    def __init__(self, regularizer : Regularizer, model : ProblemParameters, num_cols : int, transpose : bool):
        self.transpose = transpose
        self.num_cols = num_cols.type(torch.int32)
        self.regularizer_list = list()
        for i in range(self.num_cols):
            self.regularizer_list.append(type(regularizer)(model))

    def prox(self, input : torch.Tensor, eta : float) -> torch.Tensor:
        output = torch.clone(input)

        def worker(i, transpose, regularizer):
            if transpose:
                colx = input[i, :]
            else:
                colx = input[:, i]
            coly = regularizer.prox(colx, eta)
            if transpose:
                output[i, :] = coly
            else:
                output[:, i] = coly

        processes = []
        for i in range(self.num_cols):
            #p = mp.Process(target=worker, args=(i, self.transpose, self.regularizer_list[i]))
            #processes.append(p)
            #p.start()
            if transpose:
                colx = input[i, :]
            else:
                colx = input[:, i]
            coly = regularizer.prox(colx, eta)
            if transpose:
                output[i, :] = coly
            else:
                output[:, i] = coly


        #for p in processes:
        #    p.join()

        return output

    def eval(self, input_tensor):
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        def worker(i, transpose, regularizer):
            if transpose:
                col = input_tensor[i, :]
            else:
                col = input_tensor[:, i]
            value = regularizer[i].eval(col)
            with mp.Lock():
                sum_value.add_(value)

        processes = []
        for i in range(self.num_cols): 
            p = mp.Process(target=worker, args=(i, self.transpose, self.regularizer_list[i]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return sum_value

    def fenchel(self, grad1 : torch.Tensor, grad2 : torch.Tensor) -> float:
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        def worker(i, transpose, regularizer_list):
            if transpose:
                col1 = grad1[i, :]
                col2 = grad2[i, :]
            else:
                col1 = grad1[:, i]
                col2 = grad2[:, i]
            value = regularizer_list[i].fenchel(col1, col2)
            with mp.Lock():
                sum_value.add_(value)
            if transpose:
                grad1[i, :] = col1
                grad2[i, :] = col2
            else:
                grad1[:, i] = col1
                grad2[:, i] = col2

        processes = []
        for i in range(self.num_cols):  # Assuming self.num_cols is defined somewhere
            p = mp.Process(target=worker, args=(i, self.transpose, self.regularizer_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return sum_value

    def provides_fenchel(self) -> bool:
        ok = True
        for i in range(self.num_cols):
            ok = ok and self.regularizer_list[i].provides_fenchel()

        return ok

    def print(self) -> None:
        logger.info("Regularization for matrices")
        self.regularizer_list[0].print()

    def lambda_1(self) -> float:
        return self.regularizer_list[0].lambda_1()

    def lazy_prox(self, input_tensor, indices, eta):
        output = torch.clone(input_tensor)

        def worker(i, transpose, regularizer_list):
            if transpose:
                colx = input_tensor[i, :]
            else:
                colx = input_tensor[:, i]
            coly = regularizer_list[i].lazy_prox(colx, indices, eta)
            if transpose:
                output[i, :] = coly
            else:
                output[:, i] = coly

        processes = []
        for i in range(self.num_cols):
            p = mp.Process(target=worker, args=(i, self.transpose, self.regularizer_list))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output

    def is_lazy(self) -> bool:
        return self.regularizer_list[0].is_lazy()

class RegVecToMat(Regularizer):
    
    def __init__(self, regularizer : Regularizer, model : ProblemParameters):
        super().__init__(model)
        parameter_tmp = model
        parameter_tmp.verbose = False
        self.regularizer = type(regularizer)(parameter_tmp)

    def prox(self, input : torch.Tensor, eta : float) -> torch.Tensor:
        weight, _ = self.get_wb(input)
        output = self.regularizer.prox(weight, eta)
        return output

    def eval_tensor(self, input : torch.Tensor) -> float:
        weight, _ = self.get_wb(input)
        return self.regularizer.eval_tensor(weight)

    def fenchel(self, grad1 : torch.Tensor, grad2 : torch.Tensor) -> float:
        grad1_flattened = torch.flatten(grad1)
        weight, bias = self.get_wb(grad2)
        if self.intercept:
            bias_nrm_squ = torch.pow(torch.linalg.vector_norm(bias), 2)
            if bias_nrm_squ > 1e-7:
                return float("inf")
        return  self.regularizer.fenchel(grad1_flattened, weight) 

    def print(self) -> None:
        self.regularizer.print()

    def strong_convexity(self) -> float:
        return 0 if self.intercept else self.regularizer.strong_convexity()

    def lambda_1(self) -> float:
        return self.regularizer.lambda_1()

    def lazy_prox(self, input : torch.Tensor, indices : torch.Tensor, eta : float) -> None:
        weight, _ = get_wb(input);
        return self.regularizer.lazy_prox(weight, indices, eta);
    
    def is_lazy(self) -> bool:
        return self.regularizer.is_lazy()

    def get_wb(self, input : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = input.size(dim=1)
        if (self.intercept):
            weight = input[:, :p-1]
            bias = input[:, p]
        else:
            weight = input[:, :p]
            bias = None
            
        return weight, bias
