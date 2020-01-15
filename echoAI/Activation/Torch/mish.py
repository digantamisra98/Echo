"""
Applies the mish function element-wise:

.. math::

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

Reference: 
https://arxiv.org/abs/1908.08681

"""

# import pytorch
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

class MishFunction(Function):

"""
    Applies the mish function element-wise:

    .. math::

        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Plot:

    .. figure::  _static/mish.png
        :align:   center


    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    """
    if torch.cuda.is_available(): 
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            y = x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            sigmoid = torch.sigmoid(x)
            tanh_sp = torch.tanh(F.softplus(x)) 
            return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))
    else:
        @torch.jit.script
        def mish(input):
            delta = torch.exp(-input)
            alpha = 1 + 2 * delta
            return input * alpha / (alpha + 2* delta * delta)
        
