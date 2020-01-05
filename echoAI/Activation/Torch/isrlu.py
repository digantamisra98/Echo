"""
Applies the ISRLU (Inverse Square Root Linear Unit) function element-wise:

.. math::

    ISRLU(x)=\\left\\{\\begin{matrix} x, x\\geq 0 \\\\  x * (\\frac{1}{\\sqrt{1 + \\alpha*x^2}}), x <0 \\end{matrix}\\right.

ISRLU paper:
https://arxiv.org/pdf/1710.09967.pdf
"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class ISRLU(nn.Module):
    """
    Applies the ISRLU function element-wise:

    .. math::

        ISRLU(x)=\\left\\{\\begin{matrix} x, x\\geq 0 \\\\  x * (\\frac{1}{\\sqrt{1 + \\alpha*x^2}}), x <0 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/isrlu.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - alpha: hyperparameter α controls the value to which an ISRLU saturates for negative inputs (default = 1)

    References:
        - ISRLU paper: https://arxiv.org/pdf/1710.09967.pdf

    Examples:
        >>> m = ISRLU(alpha=1.0)
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
        return Func.isrlu(input, self.alpha)
