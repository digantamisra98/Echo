"""
Applies the Mila function element-wise:

.. math::

    mila(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x))

Refer to:
https://github.com/digantamisra98/Mila
"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class Mila(nn.Module):
    """
    Applies the Mila function element-wise:

    .. math::

        mila(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x)

    Plot:

    .. figure::  _static/mila.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - beta: scale to control the concavity of the global minima of the function (default = -0.25)

    References:
        -  https://github.com/digantamisra98/Mila

    Examples:
        >>> m = Mila(beta=-0.25)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self, beta=-0.25):
        """
        Init method.
        """
        super().__init__()
        self.beta = beta

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.mila(input, self.beta)
