'''
Applies the mish function element-wise:

.. math::

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

Heng's optimized implementation of mish:
https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/111457#651223
'''

# import pytorch
import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F

class MishFunction(Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid  = torch.sigmoid(x)
        softplus = F.softplus(x)
        tanh     = torch.tanh(softplus)
        return grad_output * (1-tanh*tanh)*sigmoid

class Mish(nn.Module):
    '''
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

    '''
    def forward(self, x):
        return MishFunction.apply(x)
