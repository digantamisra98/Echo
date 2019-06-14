'''
Applies the E-Swish function element-wise:

.. math::

    ESwish(x, \\beta) = \\beta*x*sigmoid(x)

See E-Swish paper:
https://arxiv.org/abs/1801.07145
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class eswish(nn.Module):
    '''
    Applies the E-Swish function element-wise:

    .. math::

        ESwish(x, \\beta) = \\beta*x*sigmoid(x)

    See E-Swish paper:
    https://arxiv.org/abs/1801.07145

    Plot:

    .. figure::  _static/eswish.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = eswish(beta=1.375)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, beta = 1.75):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta


    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.eswish(input, self.beta)
