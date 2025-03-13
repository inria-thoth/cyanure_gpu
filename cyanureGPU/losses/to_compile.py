
import torch

import torch._dynamo as dynamo

import torch_tensorrt  # noqa: F401

py_loss = torch.nn.CrossEntropyLoss()

input_data = torch.randn(2048, 2448873)
loss_labels = torch.randint(0, 204, (2448873, )).to(torch.int64)
input = torch.randn(205, 2048)


def eval_tensor(input: torch.Tensor, loss_labels: torch.Tensor,
                input_data: torch.Tensor, matmul_result: torch.Tensor = None) -> float:
    if matmul_result is not None:
        tmp = matmul_result
    else:
        tmp = torch.matmul(input, input_data)

    return py_loss(tmp.T, loss_labels)


eval_tensor_compiled = torch.compile(eval_tensor, backend="dynamo", mode="max-autotune")

explanation = dynamo.explain(eval_tensor, input, loss_labels, input_data)
print(explanation)

trt_exp_program = torch.export.export(eval_tensor_compiled, args=(input, loss_labels, input_data))

torch.export.save(trt_exp_program, "trt_model.ep")
