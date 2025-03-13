from cyanureGPU.logger import setup_custom_logger
import torch
from cyanureGPU.losses.loss_matrix import LinearLossMat

logger = setup_custom_logger("INFO")


class SquareLossMat(LinearLossMat):

    def __init__(self, data: torch.Tensor, y: torch.Tensor, intercept: bool):
        super().__init__(data, y, intercept)
        self.id = "SQUARE"

    def pre_compute(self, input: torch.Tensor) -> float:

        tmp = self.pred_tensor(input, None)

        tmp = torch.sub(tmp, self.labels)

        return tmp

    def eval_tensor(self, input: torch.Tensor, matmul_result: torch.Tensor = None, precompute: torch.Tensor = None) -> float:
        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = self.pred_tensor(input, None)
            grad1 = torch.sub(grad1, self.labels)
        else:
            grad1 = precompute
        return 0.5 * torch.linalg.norm(grad1)**2 / grad1.size(dim=1)

    def eval(self, input: torch.Tensor, i: int) -> float:
        tmp = self.pred(i, input, None)
        tmp.sub_(self.labels[:, i])
        return 0.5 * torch.linalg.norm(tmp)**2

    def print(self) -> None:
        logger.info("Square Loss is used")

    def fenchel(self, input: torch.Tensor) -> float:
        num_class = input.size(dim=1)

        # Assuming 'input' and 'self.labels' are your tensors
        input_size = input.size()

        # Define the chunk size along the last dimension
        chunk_size = 1000

        # Calculate the number of chunks
        num_chunks = input_size[-1] // chunk_size

        # Initialize the result sum
        result_sum = 0.0

        # Perform batched matrix multiplication in chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size

            # Extract submatrices
            input_chunk = input[..., start_idx:end_idx]
            labels_chunk = self.labels[..., start_idx:end_idx]

            # Perform dot product for the chunk
            dot_product_chunk = torch.tensordot(input_chunk, labels_chunk, dims=([-1], [-1]))

            # Add the result to the sum
            result_sum += dot_product_chunk

        return 0.5 * torch.linalg.norm(input)**2 / num_class + torch.sum(torch.tensor([result_sum])) / num_class

    def get_grad_aux(self, input: torch.Tensor, matmul_result: torch.Tensor = None,
                     precompute: torch.Tensor = None) -> torch.Tensor:
        if precompute is None:
            if matmul_result is not None:
                grad1 = matmul_result
            else:
                grad1 = self.pred_tensor(input, None)
            grad1 = torch.sub(grad1, self.labels)
        else:
            grad1 = precompute
        return grad1

    def scal_grad(self, input: torch.Tensor, i: int) -> torch.Tensor:
        output = self.pred(i, input, None)
        ycol = self.labels[:, i]
        return output.sub_(ycol)

    def lipschitz_constant(self) -> float:
        return 1.0

    def get_dual_constraints(self, grad1: torch.Tensor) -> torch.Tensor:
        if (self.intercept):
            grad1 = self.center_rows(grad1)
        return grad1

    # center the matrix
    def center_rows(self, grad: torch.Tensor) -> torch.Tensor:
        mean_rows = torch.mean(grad, 0)
        grad -= mean_rows

        return grad
