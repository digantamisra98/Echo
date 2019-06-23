'''
Script defined the SReLU (S-shaped Rectified Linear Activation Unit):

.. math::

    h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.

See SReLU paper:
https://arxiv.org/pdf/1512.07030.pdf
'''

# import pytorch
import torch
from torch import nn
from torch.nn.parameter import Parameter

class srelu(nn.Module):
    '''
    SReLU (S-shaped Rectified Linear Activation Unit): a combination of three linear functions, which perform mapping R → R with the following formulation:

    .. math::

        h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.

    with 4 trainable parameters.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:

        .. math:: \\{t_i^r, a_i^r, t_i^l, a_i^l\\}

    4 trainable parameters, which model an individual SReLU activation unit. The subscript i indicates that we allow SReLU to vary in different channels. Parameters can be initialized manually or randomly.

    References:
        - See SReLU paper:
        https://arxiv.org/pdf/1512.07030.pdf

    Examples:
        >>> srelu_activation = srelu((2,2))
        >>> t = torch.randn((2,2), dtype=torch.float, requires_grad = True)
        >>> output = srelu_activation(t)
    '''
    def __init__(self, in_features, parameters = None):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - parameters: (tr, tl, ar, al) parameters for manual initialization, default value is None. If None is passed, parameters are initialized randomly.
        '''
        super(srelu,self).__init__()
        self.in_features = in_features

        if parameters == None:
            self.tr = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad = True))
            self.tl = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad = True))
            self.ar = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad = True))
            self.al = Parameter(torch.randn(in_features, dtype=torch.float, requires_grad = True))
        else:
            self.tr, self.tl, self.ar, self.al = parameters

    def forward(self, x):
        '''
        Forward pass of the function
        '''
        return (x >= self.tr).float() * (self.tr + self.ar * (x + self.tr)) + (x < self.tr).float() * (x > self.tl).float() * x + (x <= self.tl).float() * (self.tl + self.al * (x + self.tl))
