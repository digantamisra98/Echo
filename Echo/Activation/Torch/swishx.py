'''
Applies the Swish-X function element-wise:
Swish-X(x, beta) = x*sigmoid(beta,x) = x/(1+e^(-beta*x))

See Swish-X paper:
https://arxiv.org/pdf/1710.05941.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class swishx(nn.Module):
    '''
    Applies the Swish-X function element-wise:
    Swish-X(x, beta) = x*sigmoid(beta,x) = x/(1+e^(-beta*x))

    See Swish paper:
    https://arxiv.org/pdf/1710.05941.pdf

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = swishx(beta=1.25)
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
        return Func.swishx(input, self.beta)
