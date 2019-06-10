'''
Applies the softplus function element-wise:
softplus(x) = ln(1 + exp(x))
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class softplus(nn.Module):
    '''
    Applies the softplus function element-wise:
    softplus(x) = ln(1 + exp(x))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = softplus()
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
        return Func.softplus(input)
