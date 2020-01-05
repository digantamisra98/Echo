"""
Applies the Sigmoid Linear Unit (SiLU) function element-wise:

.. math::

    silu(x) = x * sigmoid(x)


See related paper:
https://arxiv.org/pdf/1606.08415.pdf
"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class Silu(nn.Module):
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:

    .. math::

        silu(x) = x * sigmoid(x)

    Plot:

    .. figure::  _static/silu.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - inplace - (bool) if inplace == True operation is performed inplace

    References:
        -  Related paper:
        https://arxiv.org/pdf/1606.08415.pdf

    Examples:
        >>> m = Silu(inplace = False)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self, inplace=False):
        """
        Init method.
        """
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.silu(input, self.inplace)
