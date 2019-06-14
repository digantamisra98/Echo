'''
Applies the β mish function element-wise:
β mish(x) = x * tanh(ln((1 + exp(x))^β))
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class beta_mish(nn.Module):
    '''
    Applies the β mish function element-wise:
    β mish(x) = x * tanh(ln((1 + exp(x))^β))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = beta_mish(beta=1.5)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, beta = 1.5):
        '''
        Init method.
        '''
        super().__init__()
        self.beta = beta

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.beta_mish(input, self.beta)
