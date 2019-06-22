'''
Applies the Swish function element-wise:

.. math::

    Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

See Swish paper:
https://arxiv.org/pdf/1710.05941.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class swish(nn.Module):
    '''
    Applies the Swish function element-wise:

    .. math::

        Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

    Plot:

    .. figure::  _static/swish.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - beta: hyperparameter, which controls the shape of the bump (default = 1.25)

    References:
        - See Swish paper:
        https://arxiv.org/pdf/1710.05941.pdf

    Examples:
        >>> m = swish(beta=1.25)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, beta = 1.25):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta


    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.swish(input, self.beta)
