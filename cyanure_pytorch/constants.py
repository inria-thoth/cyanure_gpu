import torch

EPSILON = 10e-10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda"

NUMBER_OPTIM_PROCESS_INFO = 6