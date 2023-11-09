import torch
import math

from cyanure_pytorch.losses.loss import Loss

from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")

class LogisticLoss(Loss):

    def __init__(self, data : torch.Tensor, y : torch.Tensor, intercept : bool):
        super().__init__(data, y, intercept)
        self.id="LOGISTIC"

    def eval(self, input : torch.Tensor, i : int) -> float:
        res = self.labels[i] * self.pred(i,input)
        if res > 0:
            return math.log(1.0 + math.exp(-res))
        else:
            return math.log(1.0 + math.exp(res))

    def eval_tensor(self, input : torch.Tensor) -> float:
        tmp = self.pred_tensor(input)
        tmp = torch.mul(tmp, self.labels)
        tmp = torch.log(1.0 + torch.exp(torch.neg(tmp)))
        return torch.sum(tmp) / (tmp.size(dim=0)) 

    def print(self) -> None:
        logger.info("Logistic Loss is used")

    def fenchel(self, input : torch.Tensor) -> float:
        
        sum_value = 0
        n = input.size(dim=0)
        prod = torch.mul(self.labels, input)
        sum_vector = torch.special.xlogy(1.0+prod, 1.0+prod)+torch.special.xlogy(-prod, -prod)
        return torch.sum(sum_vector)/n

    def scal_grad(self, input : torch.Tensor, i : int) -> float:
        label = self.labels[i]
        ss = self.pred(i,input)
        s = -label/(1.0+math.exp(label*ss))
    
        return s

    def get_grad_aux(self, input : torch.Tensor) -> torch.Tensor:
        grad1 = self.pred_tensor(input)
        grad1 = torch.mul(grad1, self.labels)
        grad1 = 1.0 / (torch.exp(grad1) + 1.0)
        grad1 = torch.mul(-grad1, self.labels)

        return grad1

    def lipschitz_constant(self) -> float:
        return 0.25
    
    def get_dual_constraints(self, grad1 : torch.Tensor) -> torch.Tensor:
        if (self.intercept):
            grad1 = self.project_sft_binary(grad1, self.labels);
        
        return grad1

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