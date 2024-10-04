import torch
import numpy as np

EPSILON = 10e-10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMBER_OPTIM_PROCESS_INFO = 6
TENSOR_TYPE = torch.float32
ARRAY_TYPE = np.float32
