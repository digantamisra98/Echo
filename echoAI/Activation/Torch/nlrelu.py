"""
Applies the natural logarithm ReLU activation function element-wise:

Refer to: https://arxiv.org/abs/1908.03682

"""

# import pytorch
from torch import nn

# import activation functions
import echoAI.Activation.Torch.functional as Func


class NLReLU(nn.Module):
    """
    Applies the natural logarithm ReLU activation function element-wise:

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - beta: (default = 1.0)

    References:
        -  https://arxiv.org/abs/1908.03682

    Examples:
        >>> m = NLReLU(beta=1.0)
        >>> input = torch.randn(2)
        >>> output = m(input)

    """

    def __init__(self, beta=1.0):
        """
        Init method.
        """
        super().__init__()
        self.beta = beta

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return Func.nl_relu(input, self.beta)
