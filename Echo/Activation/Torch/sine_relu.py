'''
Applies the SineReLU function element-wise:

.. math::

    SineReLU(x, \\epsilon) = \\left\\{\\begin{matrix} x , x > 0 \\\\ \\epsilon * (sin(x) - cos(x)), x \\leq  0 \\end{matrix}\\right.


See related Medium article:
https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d
'''

# import pytorch
import torch
from torch import nn

# import activation functions
import Echo.Activation.Torch.functional as Func

class sine_relu(nn.Module):
    '''
    Applies the SineReLU function element-wise:

    .. math::

        SineReLU(x, \\epsilon) = \\left\\{\\begin{matrix} x , x > 0 \\\\ \\epsilon * (sin(x) - cos(x)), x \\leq  0 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/sine_relu.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - epsilon: hyperparameter (default = 0.01) used to control the wave amplitude

    References:
        - See related Medium article:
        https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d

    Examples:
        >>> m = sine_relu()
        >>> input = torch.randn(2)
        >>> output = m(input)

    '''
    def __init__(self, epsilon = 0.01):
        '''
        Init method.
        '''
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return Func.sineReLU(input, self.epsilon)
