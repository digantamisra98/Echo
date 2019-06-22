'''
Applies Soft Clipping function element-wise:

.. math::

    SC(x) = 1 / \\alpha * log(\\frac{1 + e^{\\alpha * x}}{1 + e^{\\alpha * (x-1)}})

See SC paper:
https://arxiv.org/pdf/1810.11509.pdf
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class soft_clipping(nn.Module):
    '''
    Applies the Soft Clipping function element-wise:

    .. math::

        SC(x) = 1 / \\alpha * log(\\frac{1 + e^{\\alpha * x}}{1 + e^{\\alpha * (x-1)}})

    Plot:

    .. figure::  _static/sc.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - alpha: hyper-parameter, which determines how close to linear the central region is and how sharply the linear region turns to the asymptotic values

    References:
        - See SC paper:
            https://arxiv.org/pdf/1810.11509.pdf

    Examples:
        >>> m = soft_clipping(alpha=0.5)
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, alpha = 0.5):
        '''
        Init method.
        '''
        super().__init__()
        self.alpha = alpha


    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.soft_clipping(input, self.alpha)
