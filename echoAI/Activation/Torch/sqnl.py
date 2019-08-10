'''
Applies the SQNL function element-wise:

.. math::

    SQNL(x) = \\left\\{\\begin{matrix} 1, x > 2 \\\\ x - \\frac{x^2}{4}, 0 \\leq x \\leq 2 \\\\  x + \\frac{x^2}{4}, -2 \\leq x < 0 \\\\ -1, x < -2 \\end{matrix}\\right.

See SQNL paper:
https://ieeexplore.ieee.org/document/8489043
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func

class SQNL(nn.Module):
    '''
    Applies the SQNL function element-wise:

    .. math::

        SQNL(x) = \\left\\{\\begin{matrix} 1, x > 2 \\\\ x - \\frac{x^2}{4}, 0 \\leq x \\leq 2 \\\\  x + \\frac{x^2}{4}, -2 \\leq x < 0 \\\\ -1, x < -2 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/sqnl.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        - See SQNL paper:
        https://ieeexplore.ieee.org/document/8489043

    Examples:
        >>> m = SQNL()
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
        return Func.sqnl(input)
