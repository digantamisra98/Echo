'''
Applies the Mila function element-wise:

.. math::

    \\mila(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x))
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class mila(nn.Module):
    '''
    Applies the Mila function element-wise:

    .. math::

        \\mila(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = mila(beta=-0.25)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, beta = -0.25):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.mila(input, self.beta)
