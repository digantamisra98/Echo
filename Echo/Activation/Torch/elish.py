'''
Applies the ELiSH (Exponential Linear Sigmoid SquasHing) function element-wise:

.. math::

    ELiSH(x) = \\left\\{\\begin{matrix} x / (1+e^{-x}), x \\geq 0 \\\\ (e^{x} - 1) / (1 + e^{-x}), x < 0 \\end{matrix}\\right.


See ELiSH paper:
https://arxiv.org/pdf/1808.00783.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class elish(nn.Module):
    '''
    Applies the ELiSH (Exponential Linear Sigmoid SquasHing) function element-wise:

    .. math::

        ELiSH(x) = \\left\\{\\begin{matrix} x / (1+e^{-x}), x \\geq 0 \\\\ (e^{x} - 1) / (1 + e^{-x}), x < 0 \\end{matrix}\\right.

    See ELiSH paper:
    https://arxiv.org/pdf/1710.05941.pdf

    Plot:

    .. figure::  _static/elish.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = elish()
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
        return Func.elish(input)
