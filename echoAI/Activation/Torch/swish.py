"""
Applies the Swish function element-wise:

.. math::

    Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

See Swish paper:
https://arxiv.org/pdf/1710.05941.pdf
"""

# import pytorch
import torch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class Swish(nn.Module):
    """
    Applies the Swish function element-wise:

    .. math::

        Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

    Plot:

    .. figure::  _static/swish.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        - See Swish paper:
        https://arxiv.org/pdf/1710.05941.pdf

    Examples:
        >>> m = Swish(beta=1.25)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()
        self.beta = nn.Parameter(torch.randn(1))
        self.beta.requires_grad = True

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.swish(input, self.beta)
