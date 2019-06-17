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

    .. math::

        weightedtanh(x) = tanh(x * weight)

    See additional documentation for :mod:`Echo.Activation.Torch.weightedTanh`.
    '''
    return torch.tanh(weight * input)

def mish(input):
    '''
    Applies the mish function element-wise:

    .. math::

        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    See additional documentation for :mod:`Echo.Activation.Torch.mish`.
    '''
    return input * torch.tanh(F.softplus(input))

def swish(input):
    '''
    Applies the swish function element-wise:

    .. math::

        swish(x) = x * sigmoid(x)

    See additional documentation for :mod:`Echo.Activation.Torch.swish`.
    '''
    return input * torch.sigmoid(input)


def aria2(input, beta=1, alpha=1.5):
    '''
    Applies the Aria-2 function element-wise:

    .. math::

        Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

    See additional documentation for :mod:`Echo.Activation.Torch.aria2`.
    '''
    return torch.pow((1+torch.exp(-beta * input)),-alpha)

def beta_mish(input, beta=1.5):
    '''
    Applies the β mish function element-wise:

        .. math::

            \\beta mish(x) = x * tanh(ln((1 + e^{x})^{\\beta}))

    See additional documentation for :mod:`Echo.Activation.Torch.beta_mish`.
    '''
    return input * torch.tanh(torch.log(torch.pow((1+torch.exp(input)),beta)))

def eswish(input, beta=1.75):
    '''
    Applies the E-Swish function element-wise:

        .. math::

            ESwish(x, \\beta) = \\beta*x*sigmoid(x)

    See additional documentation for :mod:`Echo.Activation.Torch.eswish`.
    '''
    return beta * input * torch.sigmoid(input)

def swishx(input, beta=1.25):
    '''
    Applies the Swish-X function element-wise:

        .. math::

            SwishX(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

    See additional documentation for :mod:`Echo.Activation.Torch.swishx`.
    '''
    return input/(1+torch.exp(-beta*input))
