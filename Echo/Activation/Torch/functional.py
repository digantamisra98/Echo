'''
Script provides functional interface for custom activation functions.
'''

# import pytorch
import torch
from torch import nn
import torch.nn.functional as F

def weighted_tanh(input, weight = 1):
    '''
    Applies the weighted tanh function element-wise:
    weighted_tanh(x) = tanh(x * weight)
    See additional documentation for weightedTanh class.
    '''
    return torch.tanh(weight * input)

def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

def swish(input):
    '''
    Applies the swish function element-wise:
    swish(x) = x * sigmoid(x)
    See additional documentation for swish class.
    '''
    return input * torch.sigmoid(input)

def softplus(input):
    '''
    Applies the softplus function element-wise:
    softplus(x) = ln(1+exp(x))
    See additional documentation for softplus class.
    '''
    return F.softplus(input)

def aria2(input, beta=1, alpha=1.5):
    '''
    Applies the softplus function element-wise:
    aria2(x) = (1+exp(-beta*x))^-alpha
    See additional documentation for aria2 class.
    '''
    return (1+torch.exp(-beta * input)) ** -alpha
