"""
Script implements soft exponential activation:

.. math::

    SoftExponential(x, \\alpha) = \\left\\{\\begin{matrix} - \\frac{log(1 - \\alpha(x + \\alpha))}{\\alpha}, \\alpha < 0\\\\  x, \\alpha = 0\\\\  \\frac{e^{\\alpha * x} - 1}{\\alpha} + \\alpha, \\alpha > 0 \\end{matrix}\\right.

See related paper:
https://arxiv.org/pdf/1602.01321.pdf
"""

# import torch
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SoftExponential(nn.Module):
    """
    Implementation of soft exponential activation:

        .. math::

            SoftExponential(x, \\alpha) = \\left\\{\\begin{matrix} - \\frac{log(1 - \\alpha(x + \\alpha))}{\\alpha}, \\alpha < 0\\\\  x, \\alpha = 0\\\\  \\frac{e^{\\alpha * x} - 1}{\\alpha} + \\alpha, \\alpha > 0 \\end{matrix}\\right.

    with trainable parameter alpha.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - alpha - trainable parameter

    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf

    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, alpha=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: learnable parameter
            aplha is initialized with zero value by default
        """
        super(SoftExponential, self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha is None:
            self.alpha = Parameter(torch.tensor(0.0))  # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.tensor(alpha))  # create a tensor out of alpha

        self.alpha.requiresGrad = True  # set requiresGrad to true!

    def forward(self, x):
        """
        Forward pass of the function
        """
        if self.alpha == 0.0:
            return x

        if self.alpha < 0.0:
            return -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if self.alpha > 0.0:
            return (torch.exp(self.alpha * x) - 1) / self.alpha + self.alpha
