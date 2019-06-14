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


def aria2(input, beta=1, alpha=1.5):
    '''
    Applies the Aria-2 function element-wise:
    aria2(x) = (1+exp(-beta*x))^-alpha
    See additional documentation for aria2 class.
    '''
    return torch.pow((1+torch.exp(-beta * input)),-alpha)

def beta_mish(input, beta=1.5):
    '''
    Applies the β mish function element-wise:
    β mish(x) = x * tanh(ln((1 + exp(x))^β))

    See additional documentation for beta_mish class.
    '''
    return input * torch.tanh(torch.log(torch.pow((1+torch.exp(input)),beta)))

def eswish(input, beta=1.75):
    '''
    Applies the E-Swish function element-wise:
    E-Swish(x, beta) = beta*x*sigmoid(x)

    See additional documentation for eswish class.
    '''
    return beta * input * torch.sigmoid(input)

def swishx(input, beta=1.25):
    '''
    Applies the Swish-X function element-wise:
    Swish-X(x, beta) = x*sigmoid(beta,x) = x/(1+e^(-beta*x))

    See additional documentation for swish class.
    '''
    return input/(1+torch.pow(e,-beta*x))
