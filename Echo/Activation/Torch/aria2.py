'''
Applies the Aria-2 function element-wise:
Aria-2(x, alpha, beta) = (1+exp(-beta*x))^-alpha

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
    aria2(x) = (1+exp(-beta*x))^-alpha

    Aria paper:
    https://arxiv.org/abs/1805.08878

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = aria2()
        >>> input = torch.randn(2)
        >>> output = m(input, beta=0.5, alpha=1)

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
        return Func.swish(input, self.alpha, self.beta)
