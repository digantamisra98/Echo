"""
Applies the Le Cun's Tanh function element-wise:

.. math::

    lecun_tanh(x) = 1.7159 * tanh((2/3) * input)

See additional documentation for :mod:`echoAI.Activation.Torch.lecun_tanh`.
"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class LeCunTanh(nn.Module):
    """
    Applies the Le Cun's Tanh function element-wise:

    .. math::

        lecun_tanh(x) = 1.7159 * tanh((2/3) * input)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Examples:
        >>> m = LeCunTanh()
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self):
        """
        Init method.
        """
        super().__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.lecun_tanh(input)
