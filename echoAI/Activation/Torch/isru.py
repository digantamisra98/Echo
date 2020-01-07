"""
Applies the ISRU (Inverse Square Root Unit) function element-wise:

.. math::

    ISRU(x) = \\frac{x}{\\sqrt{1 + \\alpha * x^2}}

ISRU paper:
https://arxiv.org/pdf/1710.09967.pdf
"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class ISRU(nn.Module):
    """
    Applies the ISRU function element-wise:

    .. math::

        ISRU(x) = \\frac{x}{\\sqrt{1 + \\alpha * x^2}}

    Plot:

    .. figure::  _static/isru.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - alpha: A constant (default = 1.0)

    References:
        - ISRU paper:
        https://arxiv.org/pdf/1710.09967.pdf

    Examples:
        >>> m = ISRU(alpha=1.0)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self, alpha=1.0):
        """
        Init method.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.isru(input, self.alpha)
