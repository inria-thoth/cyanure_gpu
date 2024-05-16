import torch

from cyanure_pytorch.losses.loss_matrix import LinearLossMat
from cyanure_pytorch.constants import DEVICE
from cyanure_pytorch.logger import setup_custom_logger

logger = setup_custom_logger("INFO")


class MultiClassLogisticLoss(LinearLossMat):

    def __init__(self, data: torch.Tensor, y: torch.Tensor, intercept: bool):
        super().__init__(data, y, intercept)
        self.n_classes = int(torch.max(y) + 1)
        self.id = 'MULTI_LOGISTIC'
        self.number_data = self.labels.shape[0]
        self.one_hot = torch.nn.functional.one_hot(self.labels.to(torch.int64), self.n_classes).T.to(torch.int32)
        self.boolean_mask = torch.zeros(self.n_classes, self.number_data, dtype=torch.bool).to(DEVICE)
        index_mask = torch.arange(self.n_classes).unsqueeze(1).expand(self.n_classes, self.number_data).to(DEVICE)
        label_mask = torch.unsqueeze(self.labels, 0).expand(self.n_classes, self.number_data)
        self.boolean_mask = torch.eq(index_mask, label_mask)
        self.loss_labels = self.labels.type(torch.LongTensor).to(DEVICE)   
    
    def pre_compute(self, input: torch.Tensor) -> float:

        tmp = self.pred_tensor(input, None)
        diff = torch.masked_select(tmp, self.boolean_mask).unsqueeze(0).expand(self.n_classes, self.number_data)

        tmp.sub_(diff)
        # Find max and perform subsequent operations
        
        mm = tmp.max(dim=0, keepdim=True).values
        tmp.sub_(mm)
                   
        tmp = tmp.exp()
        
        sum_matrix = torch.abs(tmp).sum(dim=0, keepdim=True)

        return tmp, sum_matrix, mm
    
    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> float:
        if precompute is None:
            if matmul_result is not None:
                tmp = matmul_result
            else:
                tmp = torch.matmul(input, self.input_data)
        
            diff = torch.masked_select(tmp, self.boolean_mask).unsqueeze(0).expand(self.n_classes, self.number_data)

            tmp = tmp - diff
            # Find max and perform subsequent operations
            
            mm_vector = tmp.max(dim=0, keepdim=True).values
            tmp.sub_(mm_vector)
                    
            tmp = tmp.exp()
            
            sum_matrix = torch.abs(tmp).sum(dim=0, keepdim=True)
        else:
            tmp = precompute[0]
            sum_matrix = precompute[1]  
            mm_vector = precompute[2]  

        log_sums = torch.log(sum_matrix)
        sum_value = torch.sum(mm_vector) + torch.sum(log_sums)

        return sum_value / self.number_data

    def eval(self, input: torch.Tensor, i: int) -> float:
        tmp = self.pred(i, input)
        tmp += -tmp[self.labels[i]]
        mm = torch.max(tmp)
        tmp += -mm
        tmp = torch.exp(tmp)
        return mm + torch.log(torch.sum(tmp))

    def print(self) -> None:
        logger.info("Multiclass logistic Loss is used")

    def xlogx(self, x):
        
        # Handling condition where x is greater than or equal to 1e-20
        res = x * torch.log(x)

        neg_mask = x < -1e-20
        # Handling condition where x is less than -1e-20
        res.masked_fill_(neg_mask, float('inf'))

        small_mask = x < 1e-20
        # Handling condition where x is less than 1e-20
        res.masked_fill_(small_mask, 0)   
        
        return res
   
    def fenchel(self, input: torch.Tensor) -> float:
        n = input.size(1)

        # Use advanced indexing to select the relevant elements
        # Add the condition for sum += xlogx(input[i * _nclasses + j] + 1.0)
        input += self.one_hot
        
        selected_xlogx = self.xlogx(input)
        # Sum along the second dimension (class dimension)
        sum_val = torch.sum(selected_xlogx)
        
        return sum_val / n

    def get_grad_aux2(self, col: torch.Tensor, ind: int) -> torch.Tensor:
        value = col[ind].clone()
        col -= value
        mm = torch.max(col)
        col -= mm
        col = torch.exp(col)
        col /= (torch.sum(torch.abs(col)))

        col[ind] = 0
        col[ind] = -(torch.sum(torch.abs(col)))
        return col

    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> torch.Tensor:
        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = torch.matmul(input, self.input_data)

            diff = torch.masked_select(grad1, self.boolean_mask).unsqueeze(0).expand(self.n_classes, self.number_data)

            grad1 = grad1 - diff
            # Find max and perform subsequent operations

            mm = grad1.max(dim=0, keepdim=True).values
            grad1.sub_(mm)

            grad1 = grad1.exp()

            sum_matrix = torch.abs(grad1).sum(dim=0, keepdim=True)
        else:
            grad1 = precompute[0]
            sum_matrix = precompute[1]    

        grad1 /= sum_matrix

        # Compute the mask for elements to be zeroed out
        mask = 1 - self.one_hot

        # Apply the mask to grad1
        grad1 = torch.mul(grad1, mask)

        # Compute the sum of absolute values along the first dimension
        abs_sum = torch.sum(torch.abs(grad1), dim=0, keepdim=True)

        # Compute the adjustment tensor
        adjustment = torch.mul(self.one_hot, abs_sum)

        # Subtract the adjustment tensor from grad1
        grad1.sub_(adjustment)

        return grad1

    def get_grad_aux_to_compile(self, matmul_result: torch.Tensor) -> torch.Tensor:
        grad1 = matmul_result

        # Subtract one-hot encoded vector from each element in grad1
        diff = grad1[self.labels.to(torch.int64), torch.arange(grad1.shape[1])]
        grad1 = grad1.clone()
        grad1 -= diff

        grad1.sub_(diff)
        # Find max and perform subsequent operations

        mm = grad1.max(dim=0, keepdim=True).values
        grad1.sub_(mm)
        
        grad1 = grad1.exp()

        sum_matrix = torch.abs(grad1).sum(dim=0, keepdim=True)

        grad1 /= sum_matrix

        # Compute the mask for elements to be zeroed out
        mask = 1 - self.one_hot

        # Apply the mask to grad1
        grad1 *= mask

        # Compute the sum of absolute values along the first dimension
        abs_sum = torch.sum(torch.abs(grad1), dim=0, keepdim=True)

        # Compute the adjustment tensor
        adjustment = self.one_hot * abs_sum
    
        # Subtract the adjustment tensor from grad1
        grad1.sub_(adjustment)
 
        return grad1

    def scal_grad(self, input: torch.Tensor, i: int) -> torch.Tensor:
        col = self.pred(i, input, None)
        return self.get_grad_aux2(col, int(self.labels[i]))

    def lipschitz_constant(self) -> float:
        return 0.25

    def get_dual_constraints(self, grad1: torch.Tensor) -> torch.Tensor:
        # scale grad1 by 1/Nclasses
        if (self.intercept):
            for i in range(grad1.size(0)):
                row = grad1[i, :]
                row = self.project_sft(row, self.labels, i)
                grad1[i, :] = row

        return grad1

    def project_sft(self, grad1_vector: torch.Tensor, labels: torch.Tensor, clas: int) -> torch.Tensor:
        labels_binary = torch.Tensor(grad1_vector.size(dim=0))
        labels_binary[labels == clas] = 1.0
        labels_binary[labels != clas] = -1.0
        return self.project_sft_binary(grad1_vector, labels_binary)

    def project_sft_binary(self, grad1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
    def l1project(self, input: torch.Tensor, thrs: float, simplex: bool = False) -> torch.Tensor:

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

            for i in range(1, sizeU):
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

        output[output > lambda_1] = output - lambda_1
        output[output < (-lambda_1)] = output + lambda_1
        output[-lambda_1 < output < lambda_1] = 0

        return output
