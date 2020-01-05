"""
Script defined the APL (ADAPTIVE PIECEWISE LINEAR UNITS):

.. math::

    APL(x_i) = max(0,x) + \\sum_{s=1}^{S}{a_i^s * max(0, -x + b_i^s)}

See APL paper:
https://arxiv.org/pdf/1412.6830.pdf
"""

# import torch
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter


class apl_function(Function):
    """
    Implementation of APL (ADAPTIVE PIECEWISE LINEAR UNITS) activation function:

        .. math::

            APL(x_i) = max(0,x) + \\sum_{s=1}^{S}{a_i^s * max(0, -x + b_i^s)}

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Arguments:
        - a: variables control the slopes of the linear segments
        - b: variables determine the locations of the hinges

    References:
        - See APL paper:
        https://arxiv.org/pdf/1412.6830.pdf

    Examples:
        >>> apl_func = apl_function.apply
        >>> t = torch.tensor([[1.,1.],[0.,-1.]])
        >>> t.requires_grad = True
        >>> S = 2
        >>> a = torch.tensor([[[1.,1.],[1.,1.]],[[1.,1.],[1.,1.]]])
        >>> b = torch.tensor([[[1.,1.],[1.,1.]],[[1.,1.],[1.,1.]]])
        >>> t = apl_func(t, a, b)
    """

    # both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, a, b):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, a, b)  # save for backward pass

        S = a.shape[0]  # get S (number of hinges)

        output = input.clamp(min=0)
        for s in range(S):
            t = -input + b[s]
            output += a[s] * t.clamp(min=0)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # set output to None
        grad_input = None
        grad_a = None
        grad_b = None

        input, a, b = ctx.saved_tensors  # restore input from context
        S = a.shape[0]  # get S (number of hinges)

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = (input >= 0).float() * grad_output
            for s in range(S):
                grad_input += (input >= 0).float() * (-a[s]) * grad_output

        if ctx.needs_input_grad[1]:
            grad_a = torch.zeros(a.size())
            for s in range(S):
                grad_as = (input >= 0).float() * (-input) * grad_output
                grad_a[s] = grad_as.sum(dim=0, keepdim=True)

        if ctx.needs_input_grad[2]:
            grad_b = torch.zeros(b.size())
            for s in range(S):
                grad_bs = (input >= 0).float() * a[s] * grad_output
                grad_b[s] = grad_bs.sum(dim=0, keepdim=True)

        return grad_input, grad_a, grad_b


class APL(nn.Module):
    """
    Implementation of APL (ADAPTIVE PIECEWISE LINEAR UNITS) unit:

        .. math::

            APL(x_i) = max(0,x) + \\sum_{s=1}^{S}{a_i^s * max(0, -x + b_i^s)}

    with trainable parameters a and b, parameter S should be set in advance.

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    Parameters:
        - S: hyperparameter, number of hinges to be set in advance
        - a: trainable parameter, control the slopes of the linear segments
        - b: trainable parameter, determine the locations of the hinges

    References:
        - See APL paper:
        https://arxiv.org/pdf/1412.6830.pdf

    Examples:
        >>> a1 = apl(256, S = 1)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, S, a=None, b=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - S (int): number of hinges
            - a - value for initialization of parameter, which controls the slopes of the linear segments
            - b - value for initialization of parameter, which determines the locations of the hinges
            a, b are initialized randomly by default
        """
        super(APL, self).__init__()
        self.in_features = in_features
        self.S = S

        # initialize parameters
        if a is None:
            self.a = Parameter(
                torch.randn((S, in_features), dtype=torch.float, requires_grad=True)
            )
        else:
            self.a = a

        if b is None:
            self.b = Parameter(
                torch.randn((S, in_features), dtype=torch.float, requires_grad=True)
            )
        else:
            self.b = b

    def forward(self, x):
        """
        Forward pass of the function
        """
        output = x.clamp(min=0)
        for s in range(self.S):
            t = -x + self.b[s]
            output += self.a[s] * t.clamp(min=0)

        return output
