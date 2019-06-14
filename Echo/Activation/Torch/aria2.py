'''
Applies the Aria-2 function element-wise:

.. math::

    Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

See Aria paper:
https://arxiv.org/abs/1805.08878
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class aria2(nn.Module):
    '''
    Applies the Aria-2 function element-wise:

    .. math::

        Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

    Aria paper:
    https://arxiv.org/abs/1805.08878

    Plot:

    .. figure::  _static/aria2.png
        :align:   center


    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = aria2(beta=0.5, alpha=1)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, beta = 1, alpha = 1.5):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta
        self.alpha = alpha


    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.aria2(input, self.alpha, self.beta)
