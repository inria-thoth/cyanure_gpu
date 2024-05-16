import torch


def solve_binomial(a: float, b: float, c: float) -> float:

    delta = b*b-4*a*c

    # returns single largest solution, assuming delta > 0
    return (-b + torch.sqrt(delta))/(2*a)
