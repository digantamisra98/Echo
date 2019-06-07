'''
Script provides functional interface for custom activation functions.
'''

# import pytorch
import torch
from torch import nn

def weighted_tanh(input, weight = 1):
    '''
    Applies the weighted tanh function element-wise:
    f(x) = tanh(x * weight)

    See additional documentation for weightedTanh class.
    '''
    return torch.tanh(weight * input)
