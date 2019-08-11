'''
Applies the mish function element-wise:

.. math::

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func

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

    Arguments:
        - inplace: (bool) perform the operation in-place

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, inplace = False):
        '''
        Init method.
        '''
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.mish(input, inplace = self.inplace)
