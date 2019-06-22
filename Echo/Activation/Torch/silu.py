'''
Applies the Sigmoid Linear Unit (SiLU) function element-wise:

.. math::

    silu(x) = x * sigmoid(x)


See related paper:
https://arxiv.org/pdf/1606.08415.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class silu(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:

    .. math::

        silu(x) = x * sigmoid(x)

    Plot:

    .. figure::  _static/silu.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf

    Examples:
        >>> m = silu()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.swish(input)
