import torch

from cyanure_pytorch.losses.loss_matrix import LinearLossMat
from cyanure_pytorch.constants import DEVICE
from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class MultiClassLogisticLoss(LinearLossMat):

    def __init__(self, data : torch.Tensor, y : torch.Tensor, intercept : bool):
        super().__init__(data, y, intercept)
        self.n_classes = torch.max(y) + 1
        self.id = 'MULTI_LOGISTIC'
    

    def eval_tensor(self, input : torch.Tensor) -> float:
        tmp = self.pred_tensor(input, None)
        n = tmp.size(1)

        labels_line_vector = self.labels.t().to(torch.int64)
        tmp -= tmp[labels_line_vector, torch.arange(len(labels_line_vector))].unsqueeze(0).expand(tmp.shape[0], -1)

        # Calculate the maximum value along the column dimension and create mm_matrix
        mm_vector, _ = torch.max(tmp, dim=0)
        tmp.sub_(mm_vector.unsqueeze(0).expand(tmp.shape[0], -1))
        tmp.exp_()
        
        # Calculate log_sums and sum_value
        log_sums = torch.log(torch.sum(torch.abs(tmp), dim=0))
        sum_value = torch.sum(mm_vector) + torch.sum(log_sums)

        return sum_value / n

    def eval(self, input : torch.Tensor, i : int) -> float:
        tmp = self.pred(i, input)
        tmp += -tmp[self.labels[i]]
        mm = torch.max(tmp)
        tmp += -mm
        tmp = torch.exp(tmp)
        return mm + torch.log(torch.sum(tmp))

    def print(self) -> None:
        logger.info("Multiclass logistic Loss is used")

    def xlogx(self, x):
        x = x * torch.log(x)
        return x
    
    def fenchel(self, input : torch.Tensor) -> float:
        n = input.size(1)

        # Use advanced indexing to select the relevant elements
        # Add the condition for sum += xlogx(input[i * _nclasses + j] + 1.0)      
        input += torch.nn.functional.one_hot(self.labels.to(torch.int64), int(self.n_classes.item())).T
        selected_xlogx = self.xlogx(input)
        
        # Sum along the second dimension (class dimension)
        sum_val = torch.sum(selected_xlogx)
        
        return sum_val / n
    
    def kahan_sum(self, input_tensor):
        s = 0
        c = 0
        for value in input_tensor.view(-1):
            y = value - c
            t = s + y
            c = (t - s) - y
            s = t
        return s

    def get_grad_aux2(self, col : torch.Tensor, ind : int) -> torch.Tensor:
        col -= col[ind]
        mm = torch.max(col)
        col -= mm
        col = torch.exp(col)
        col /= (torch.sum(torch.abs(col)))

        col[ind] = 0        
        col[ind] = -(torch.sum(torch.abs(col)))
        return col

    def get_grad_aux(self, input : torch.Tensor) -> torch.Tensor:
        grad1 = self.pred_tensor(input, None)
        labels = self.labels

        # Subtract one-hot encoded vector from each element in grad1
        diff = grad1[labels.to(torch.int64), torch.arange(grad1.shape[1])]
        grad1 = grad1.clone()
        grad1 -= diff
        
        # Find max and perform subsequent operations
        mm = grad1.max(dim=0, keepdim=True).values
        grad1 -= mm        
        grad1 = grad1.exp()
        sum_matrix = torch.abs(grad1).sum(dim=0, keepdim=True)
        grad1 /= sum_matrix
        
        # Zero out and set the values at labels positions
        grad1 = grad1 * (1 - torch.nn.functional.one_hot(self.labels.to(torch.int64), int(self.n_classes.item())).T) 
        grad1 -= torch.nn.functional.one_hot(self.labels.to(torch.int64), int(self.n_classes.item())).T * torch.sum(torch.abs(grad1), dim=0, keepdim=True)
        
        return grad1
    
    def scal_grad(self, input : torch.Tensor, i : int, col : torch.Tensor) -> torch.Tensor:
        col = self.pred(i, input)
        return self.get_grad_aux2(col, self.labels[i])

    def lipschitz_constant(self) -> float:
        return 0.25

    def get_dual_constraints(self, grad1 : torch.Tensor) -> torch.Tensor:
        # scale grad1 by 1/Nclasses
        if (self.intercept):
            for i in range(grad1.size(0)):
                row = grad1[i, :]
                row = self.project_sft(row, self.labels, i)
                grad1[i, :] = row

        return grad1
   
    def project_sft(self, grad1_vector : torch.Tensor, labels : torch.Tensor, clas : int) -> torch.Tensor:
        labels_binary = torch.Tensor(grad1_vector.size(dim=0))
        labels_binary[labels == clas] = 1.0 
        labels_binary[labels != clas] = -1.0
        return self.project_sft_binary(grad1_vector, labels_binary)

    def project_sft_binary(self, grad1 : torch.Tensor, y : torch.Tensor) -> torch.Tensor:
        mean = torch.mean(grad1)
        n = grad1.size(dim=0)
        ztilde = torch.Tensor(n)
        count = 0
        if (mean > 0):
            for ii in range(n):
                if (y[ii] > 0):
                    count += 1
                    ztilde[ii] = grad1[ii] + 1.0
                else:
                    ztilde[ii] = grad1[ii]
            xtilde = self.l1project(ztilde, count)
            for ii in range(n):
                grad1[ii] = xtilde[ii] - 1.0 if y[ii] > 0 else xtilde[ii]
        else:
            for ii in range(n):
                if (y[ii] > 0):
                    ztilde[ii] = -grad1[ii]
                else:
                    count += 1
                    ztilde[ii] = -grad1[ii] + 1.0
            xtilde = self.l1project(ztilde, count)
            for ii in range(n):
                grad1[ii] = -xtilde[ii] if y[ii] > 0 else -xtilde[ii] + 1.0

        return grad1

    # Vectors
    def l1project(self, input : torch.Tensor, thrs : float, simplex : bool = False) -> torch.Tensor:

        output = torch.clone(input)
        if (simplex):
            output[output < 0] = 0
        else:
            output = abs(output)

        norm1 = torch.sum(output)
        if (norm1 <= thrs):
            if (not simplex):
                output = torch.clone(input)
            return None

        prU = output
        sizeU = input.size(dim=0)

        sum_value = 0
        sum_card = 0

        while (sizeU > 0):
            # put the pivot in prU[0]
            tmp = prU[0]
            prU[0] = prU[sizeU / 2]
            prU[sizeU / 2] = tmp
            pivot = prU[0]
            sizeG = 1
            sumG = pivot

            for i in range (1, sizeU):
                if (prU[i] >= pivot):
                    sumG += prU[i]
                    tmp = prU[sizeG]
                    prU[sizeG] = prU[i]
                    prU[i] = tmp
                    sizeG += 1

            if (sum_value + sumG - pivot * (sum_card + sizeG) <= thrs):
                sum_card += sizeG
                sum_value += sumG
                prU += sizeG
                sizeU -= sizeG
            else:
                prU += 1
                sizeU = sizeG - 1

        lambda_1 = (sum_value - thrs) / sum_card
        output = torch.clone(input)

        if (simplex):
            output[output < 0] = 0

        output[output>lambda_1] = output - lambda_1
        output[output<(-lambda_1)] = output + lambda_1
        output[-lambda_1<output<lambda_1] = 0

        return output