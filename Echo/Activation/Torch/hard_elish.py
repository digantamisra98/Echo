'''
Applies the HardELiSH function element-wise:

.. math::

    HardELiSH(x) = \\left\\{\\begin{matrix} x \\times max(0, min(1, (x + 1) / 2)), x \\geq 0 \\\\ (e^{x} - 1)\\times max(0, min(1, (x + 1) / 2)), x < 0 \\end{matrix}\\right.

See HardELiSH paper:
https://arxiv.org/pdf/1808.00783.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class hard_elish(nn.Module):
    '''
    Applies the HardELiSH function element-wise:

    .. math::

        HardELiSH(x) = \\left\\{\\begin{matrix} x \\times max(0, min(1, (x + 1) / 2)), x \\geq 0 \\\\ (e^{x} - 1)\\times max(0, min(1, (x + 1) / 2)), x < 0 \\end{matrix}\\right.

    See HardELiSH paper:
    https://arxiv.org/pdf/1710.05941.pdf

    Plot:

    .. figure::  _static/hard_elish.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = hard_elish()
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
        return Func.hard_elish(input)
