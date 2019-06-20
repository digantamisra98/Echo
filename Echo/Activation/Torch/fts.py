'''
Applies the FTS (Flatten T-Swish) function element-wise:

.. math::

    FTS(x) = \\left\\{\\begin{matrix} \\frac{x}{1 + e^{-x}} , x \\geq  0 \\\\ 0, x < 0 \\end{matrix}\\right.

See Flatten T-Swish paper:
https://arxiv.org/pdf/1812.06247.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class fts(nn.Module):
    '''
    Applies the FTS (Flatten T-Swish) function element-wise:

    .. math::

        FTS(x) = \\left\\{\\begin{matrix} \\frac{x}{1 + e^{-x}} , x \\geq  0 \\\\ 0, x < 0 \\end{matrix}\\right.

    See Flatten T-Swish paper:
    https://arxiv.org/pdf/1812.06247.pdf

    Plot:

    .. figure::  _static/fts.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = fts()
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
        return Func.fts(input)
