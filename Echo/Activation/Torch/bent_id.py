'''
Applies the Bent's Identity function element-wise:

.. math::

    bentId(x) = x + \\frac{\\sqrt{x^{2}+1}-1}{2}
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class bent_id(nn.Module):
    '''
    Applies the Bent's Identity function element-wise:

    .. math::

        bentId(x) = x + \\frac{\\sqrt{x^{2}+1}-1}{2}

    Plot:

    .. figure::  _static/bent_id.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = bent_id()
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
        return Func.bent_id(input)
