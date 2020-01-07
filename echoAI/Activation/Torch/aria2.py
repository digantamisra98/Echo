"""
Applies the Aria-2 function element-wise:

.. math::

    Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

See Aria paper:
https://arxiv.org/abs/1805.08878
"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class Aria2(nn.Module):
    """
    Applies the Aria-2 function element-wise:

    .. math::

        Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

    Plot:

    .. figure::  _static/aria2.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - alpha: hyper-parameter which has a two-fold effect; it reduces the curvature in 3rd quadrant as well as increases the curvature in first quadrant while lowering the value of activation (default = 1)

        - beta: the exponential growth rate (default = 0.5)

    References:
        - See Aria paper:
            https://arxiv.org/abs/1805.08878

    Examples:
        >>> m = Aria2(beta=0.5, alpha=1)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self, beta=1.0, alpha=1.5):
        """
        Init method.
        """
        super().__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.aria2(input, self.beta, self.alpha)
