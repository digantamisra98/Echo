'''
Applies the Log-Sigmoid function element-wise:
logsigmoid(x) = log(sigmoid(x)) = log(1/(1+exp(-x)))
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class logsigmoid(nn.Module):
    '''
    Applies the Log-Sigmoid function element-wise:
    logsigmoid(x) = log(sigmoid(x)) = log(1/(1+exp(-x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = logsigmoid()
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
        return Func.logsigmoid(input)
