'''
Applies the ISRU (Inverse Square Root Unit) function element-wise:

.. math::

    isru(x) = x / ((1 + \\alpha * x)^0.5)
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class isru(nn.Module):
    '''
    Applies the ISRU function element-wise:

    .. math::

        isru(x) = x / ((1 + \\alpha * x)^0.5)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = isru(alpha=1.0)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, alpha = 1.0):
        '''
        Init method.
        '''
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.isru(input, self.alpha)
