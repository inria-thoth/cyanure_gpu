from cyanure_pytorch.logger import setup_custom_logger
import torch
from typing import Tuple
import abc

from cyanure_pytorch.losses.loss import Loss

logger = setup_custom_logger("INFO")


class LinearLossMat(Loss):

    def __init__(self, data : torch.Tensor, y : torch.Tensor, intercept : bool):
        super().__init__(data, y, intercept)
        self.ones = torch.ones(self.input_data.size(dim=1))

    def add_grad(self, input : torch.Tensor, i : int, a : float = 1.0) -> torch.Tensor:
        sgrad = self.scal_grad(input,i)
        return self.add_dual_pred(i, sgrad, a)

    def double_add_grad(self, input1 : torch.Tensor, input2 : torch.Tensor, i : int, eta1 : float = 1.0, eta2 : float = -1.0, dummy : float = 1.0) -> torch.Tensor:
        sgrad1 = self.scal_grad(input1,i)
        sgrad2 = self.scal_grad(input2,i)
        sgrad1 = sgrad2 * eta2 + sgrad1 * eta1
        return self.add_dual_pred(i, sgrad1)

    def transpose(self) -> bool: 
        return True

    def add_feature_tensor(self, input : torch.Tensor, input2 : torch.Tensor, s : float) -> torch.Tensor:
        return self.add_dual_pred_tensor(input, input2, s, 1.0)
    
    def add_feature(self, i : int, s : torch.Tensor, input2 : torch.Tensor) -> torch.Tensor:
        return self.add_dual_pred(i, s, input2, 1.0, 1.0)
  
    # _X  is  p x n
    # input is nclass x p
    # output is nclass x n
    def pred_tensor(self, input : torch.Tensor, input2 : torch.Tensor) -> torch.Tensor:
        if (self.intercept):
            weight, bias = self.get_wb(input)
            if input2 is not None:
                output = torch.matmul(weight, self.input_data) + input2
            else:
                output = torch.matmul(weight, self.input_data)
            output = output + torch.outer(bias, self.ones)
        else:
            if input2 is not None:
                output = torch.matmul(input, self.input_data) + input2
            else:
                output = torch.matmul(input, self.input_data)
        
        return output
            

    def pred(self, ind : int, input : torch.Tensor, input2 : torch.Tensor) -> torch.Tensor:
        col = self.input_data[:, ind]
        if (self.intercept):
            weight, bias = self.get_wb(input)
            output = torch.matmul(weight, col) + input2
            output = output + self.scale_intercept * bias
        else:
            output = torch.matmul(input, col) + input2

        return output

    def add_dual_pred_tensor(self, input : torch.Tensor, input2 : torch.Tensor, a1 : float = 1.0, a2 : float = 1.0) -> torch.Tensor:
        if (self.intercept):
            output = torch.Tensor(input.size(dim=0), self.input_data.size(dim=0) + 1)
            weight, bias = self.get_wb(output)
            #  W = input * X.T =  (X* input.T).T
            weight = a2 * weight + a1 * torch.matmul(input, torch.transpose(self.input_data, 0, 1))
            bias = bias * a2 + a1 * self.ones
            output = torch.cat((weight, bias), dim=1)
        else:
            if input2 is not None:
                output = a2 * input2 + a1 * torch.matmul(input, torch.transpose(self.input_data, 0, 1))
            else:
                output = a1 * torch.matmul(input, self.input_data.t())
        return output

    def add_dual_pred(self, ind : int , input : torch.Tensor, input2 : torch.Tensor, a : float = 1.0, b : float = 1.0) -> torch.Tensor:
        col = self.input_data[:, ind]
        if (b != 1.0):
            input2 *= b
        if (self.intercept):
            output = torch.Tensor(input.size(dim=1), self.input_data.size(dim=0) + 1)
            # Weight and bias are zeros 
            weight, bias = self.get_wb(output)
            weight = weight + a * torch.outer(input, col)
            bias = bias + a*self.scale_intercept * input
            output = torch.cat((weight, bias), dim=1)
        else:
            output = input2 + a * self.scale_intercept * input

        return output

    def get_wb(self, input : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = input.size(dim=1)
        weight = input[:, : p-1]
        bias = input[:, p-1]

        return weight, bias

    @abc.abstractmethod
    def scal_grad(self, input : torch.Tensor, i : int)-> torch.Tensor:
        return


class LossMat(LinearLossMat):

    def __init__(self, loss : Loss):
         self.loss_list = list()
         self.data_list = list()
         self.n = loss.labels.size(dim=1)
         self.num_class = loss.labels.size(dim=0)
         self.labels_transpose = torch.transpose(loss.labels, 0, 1)
         for i in range(self.num_class):
            self.data_list[i] = loss.input_data
            ycol = self.labels_transpose[:, i]
            self.loss_list[i] = type(loss)(self.data_list[i], ycol, loss.intercept)
         self.id = self.loss_list[0].id

    def eval_tensor(self, input_tensor : torch.Tensor) -> float:
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        def worker(ii, loss_list, input_tensor):
            col = input_tensor[:, ii]
            value = loss_list[ii].eval(col)
            with mp.Lock():
                sum_value.add_(value)

        processes = []
        for ii in range(self.num_class):  # Assuming _N is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return sum_value

    def eval(self, input_tensor : torch.Tensor, i : int) -> float:
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        def worker(ii, loss_list, input_tensor, i):
            col = input_tensor[:, ii]
            value = loss_list[ii].eval(col, i)
            with mp.Lock():
                sum_value.add_(value)

        processes = []
        for ii in range(self.num_class):  # Assuming _N is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return sum_value

    def add_grad(self, input_tensor : torch.Tensor, i : int, eta : float = 1.0) -> torch.Tensor:
        output = torch.zeros_like(input_tensor)

        def worker(ii, loss_list, input_tensor, output, i, eta):
            col_result = loss_list[ii].add_grad(input_tensor[:, ii], i, eta)
            with mp.Lock():
                output[:, ii] = col_result

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor, output, i, eta))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output

    def double_add_grad(self, input1 : torch.Tensor, input2 : torch.Tensor, i : int, eta1 : float = 1.0, eta2 : float = -1.0, dummy : float = 1.0) -> torch.Tensor:
        # Input tensor is a vector
        output = torch.Tensor(input1.size())

        def worker(ii, loss_list, input_tensor, output, i):
            col_result = loss_list[ii].double_add_grad(col_input1, col_input2, i, eta1, eta2, dummy)
            with mp.Lock():
                output[:, ii] = col_result

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor, output, i, eta1, eta2, dummy))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output

    def grad(self, input_tensor : torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(input_tensor)

        def worker(ii, loss_list, input_tensor, output):
            col_result = loss_list[ii].grad(input_tensor[:, ii])
            with mp.Lock():
                output[:, ii] = col_result

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor, output))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output

    def print(self) -> None:
        logger.info("Loss for matrices")
        self.loss_list[0].print()

    def provides_fenchel(self) -> bool:
        return self.loss_list[0].provides_fenchel()

    def get_dual_variable(self, input : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input tensor is a vector
        grad1 = torch.Tensor((self.n, input.size(dim=1)))
        grad2 = torch.Tensor(input.size())

        def worker(ii, loss_list, input_tensor, grad1, grad2):
            grad1_col, grad2_col = loss_list[ii].get_dual_variable(input_tensor[:, ii])
            with mp.Lock():
                grad1[:, ii] = grad1_col
                grad2[:, ii] = grad2_col

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor, grad1, grad2))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return grad1, grad2

    def fenchel(self, input_tensor : torch.Tensor) -> float:
        sum_value = torch.tensor(0.0)  # Initialize as a PyTorch tensor

        def worker(ii, loss_list, input_tensor):
            col = input_tensor[:, ii]
            value = loss_list[ii].fenchel(col)
            with mp.Lock():
                sum_value.add_(value)

        processes = []
        for ii in range(self.num_class):  # Assuming _N is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return sum_value

    def lipschitz(self) -> float:
        return self.loss_list[0].lipschitz()

    def lipschitz_li(self, Li : torch.Tensor) -> torch.Tensor:
        return self.loss_list[0].lipschitz_li(Li)

    # input; nclass x n
    # output: p x nclass
    def add_feature_tensor(self, input : torch.Tensor, input2 : torch.Tensor, s : float) -> torch.Tensor:
        output = torch.Tensor(input.size())

        def worker(ii, loss_list, input, input2, output, s):
            col_result = loss_list[ii].add_feature_tensor(input[ii, :], input2[:, ii], s)
            with mp.Lock():
                output[:, ii] = col_result

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input, input2, output, s))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output
    
    def add_feature(self, i : int, s : torch.Tensor, input2 : torch.Tensor) -> torch.Tensor:
        # Input tensor is a vector
        output = torch.Tensor((self.loss_list[0].input_data.size(dim=0) , self.num_class))

        def worker(ii, loss_list, s, input2, output, i):
            col_result = loss_list[ii].add_feature(i, s[ii], input2[:, ii])
            with mp.Lock():
                output[:, ii] = col_result

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, s, input2, output, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output

    def scal_grad(self, input_tensor : torch.Tensor, i : int) -> torch.Tensor:
        output = torch.Tensor(self.num_class)

        def worker(ii, loss_list, input_tensor, output, i):
            col_result = loss_list[ii].scal_grad(input_tensor[:, ii], i)
            with mp.Lock():
                output[ii] = col_result

        processes = []
        for ii in range(self.num_class):  # Assuming self.num_class is defined somewhere
            p = mp.Process(target=worker, args=(ii, self.loss_list, input_tensor, output, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        return output

    def transpose(self) -> bool: 
        return False

    def get_grad_aux(self, input : torch.Tensor) -> None:
        logger.error("Not used")

    def lipschitz_constant(self) -> float:
        logger.error("Not used")
        return 0;

    def get_dual_constraints(self, grad1 : torch.Tensor)-> None:
        logger.error("Not used")
      
