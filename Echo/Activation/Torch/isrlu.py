'''
Applies the ISRLU (Inverse Square Root Linear Unit) function element-wise:

ISRLU paper:
https://arxiv.org/pdf/1710.09967.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class isrlu(nn.Module):
    '''
    Applies the ISRLU function element-wise:

    ISRLU paper:
    https://arxiv.org/pdf/1710.09967.pdf

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = isrlu(alpha=1.0)
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
        return Func.isrlu(input, self.alpha)
