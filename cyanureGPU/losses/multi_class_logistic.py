import torch

from cyanureGPU.losses.loss_matrix import LinearLossMat
from cyanureGPU.constants import DEVICE
from cyanureGPU.logger import setup_custom_logger

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

    def pre_compute(self, input: torch.Tensor) -> torch.Tensor:
        # Get prediction tensor
        tmp = self.pred_tensor(input, None)

        # Compute the difference with masked select and broadcasting
        diff = torch.masked_select(tmp, self.boolean_mask).unsqueeze(0).expand(self.n_classes, self.number_data)

        # Subtract difference (avoid in-place operations)
        tmp = tmp - diff

        # Apply log-sum-exp trick to improve numerical stability
        mm = tmp.max(dim=0, keepdim=True).values
        tmp = tmp - mm
        tmp = tmp.exp()

        # Sum matrix along the first dimension, ensuring numerical stability with epsilon
        sum_matrix = tmp.sum(dim=0, keepdim=True)

        return tmp, sum_matrix, mm

    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> float:
        if precompute is None:
            if matmul_result is not None:
                tmp = matmul_result
            else:
                tmp = self.pred_tensor(input, None)

            diff = torch.masked_select(tmp, self.boolean_mask).unsqueeze(0).expand(self.n_classes, self.number_data)

            tmp = tmp - diff
            # Find max and perform subsequent operations

            mm_vector = tmp.max(dim=0, keepdim=True).values
            tmp.sub_(mm_vector)

            tmp = tmp.exp()

            sum_matrix = tmp.sum(dim=0, keepdim=True)
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
        col = col - value
        mm = torch.max(col)
        col = col - mm
        col = torch.exp(col)
        col = col * (1 / (torch.sum(torch.abs(col))))

        col[ind] = 0
        col[ind] = -(torch.sum(torch.abs(col)))
        return col

    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None,
                     precompute: torch.Tensor = None) -> torch.Tensor:

        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = self.pred_tensor(input, None)

            diff = torch.masked_select(grad1, self.boolean_mask).unsqueeze(0).expand(self.n_classes, self.number_data)
            grad1 = grad1 - diff

            # Apply log-sum-exp trick
            mm = grad1.max(dim=0, keepdim=True).values
            grad1 = (grad1 - mm).exp()

            sum_matrix = grad1.sum(dim=0, keepdim=True)
        else:
            grad1 = precompute[0]
            sum_matrix = precompute[1]

        grad1 = grad1 / (sum_matrix)  # More stable division

        # Apply the mask to zero out certain elements
        grad1 = torch.where(self.one_hot.bool(), torch.tensor(0.0, device=grad1.device, dtype=grad1.dtype), grad1)

        # Compute the sum of absolute values along the first dimension
        abs_sum = torch.sum(torch.abs(grad1), dim=0, keepdim=True)

        # Compute the adjustment tensor
        adjustment = self.one_hot.float() * abs_sum

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
        if self.intercept:
            for i in range(grad1.size(0)):
                grad1[i, :] = self.project_sft(grad1[i, :], self.labels, i)
        return grad1

    def project_sft(self, grad1_vector: torch.Tensor, labels: torch.Tensor, clas: int) -> torch.Tensor:
        labels_binary = torch.where(labels == clas, 1.0, -1.0)
        return self.project_sft_binary(grad1_vector, labels_binary)

    def project_sft_binary(self, grad1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(grad1)

        if mean > 0:
            ztilde = grad1 + torch.where(y > 0, 1.0, 0.0)
            count = torch.sum(y > 0).item()
            xtilde = self.l1project(ztilde, count)
            grad1 = xtilde - torch.where(y > 0, 1.0, 0.0)
        else:
            ztilde = torch.where(y > 0, -grad1, -grad1 + 1.0)
            count = torch.sum(y <= 0).item()
            xtilde = self.l1project(ztilde, count)
            grad1 = torch.where(y > 0, -xtilde, -xtilde + 1.0)

        return grad1

    def l1project(self, input: torch.Tensor, thrs: float, simplex: bool = False) -> torch.Tensor:
        if simplex:
            output = torch.clamp(input, min=0)
        else:
            output = input.abs()

        norm1 = torch.sum(output)
        if norm1 <= thrs:
            return input if not simplex else output

        # Sort the input tensor in descending order
        sorted_output, _ = torch.sort(output, descending=True)

        # Calculate the cumulative sum
        cumulative_sum = torch.cumsum(sorted_output, dim=0) - thrs

        # Find rho, which is the largest index where the condition holds
        tmp = sorted_output * torch.arange(1, sorted_output.size(0) + 1, device=input.device) > cumulative_sum
        rho = torch.nonzero(tmp, as_tuple=True)[0].max()

        # Calculate the threshold lambda
        lambda_1 = cumulative_sum[rho] / (rho + 1)

        # Threshold the input tensor
        output = input.sign() * torch.clamp(output - lambda_1, min=0)

        return output
