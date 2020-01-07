"""
Script defined the BReLU (Bipolar Rectified Linear Activation Unit):

.. math::

    BReLU(x_i) = \\left\\{\\begin{matrix} f(x_i), i \\mod 2 = 0\\\\  - f(-x_i), i \\mod 2 \\neq  0 \\end{matrix}\\right.

See BReLU paper:
https://arxiv.org/pdf/1709.04054.pdf
"""

# import torch
from torch.autograd import Function


class BReLU(Function):
    """
    Implementation of BReLU activation function:

        .. math::

            BReLU(x_i) = \\left\\{\\begin{matrix} f(x_i), i \\mod 2 = 0\\\\  - f(-x_i), i \\mod 2 \\neq  0 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/brelu.png
        :align:   center

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        - See BReLU paper:
        https://arxiv.org/pdf/1709.04054.pdf

    Examples:
        >>> brelu_activation = brelu.apply
        >>> t = torch.randn((5,5), dtype=torch.float, requires_grad = True)
        >>> t = brelu_activation(t)
    """

    # both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)  # save input for backward pass

        # get lists of odd and even indices
        input_shape = input.shape[0]
        even_indices = [i for i in range(0, input_shape, 2)]
        odd_indices = [i for i in range(1, input_shape, 2)]

        # clone the input tensor
        output = input.clone()

        # apply ReLU to elements where i mod 2 == 0
        output[even_indices] = output[even_indices].clamp(min=0)

        # apply inversed ReLU to inversed elements where i mod 2 != 0
        output[odd_indices] = (
            0 - output[odd_indices]
        )  # reverse elements with odd indices
        output[odd_indices] = -output[odd_indices].clamp(min=0)  # apply reversed ReLU

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_input = None  # set output to None

        (input,) = ctx.saved_tensors  # restore input from context

        # check that input requires grad
        # if not requires grad we will return None to speed up computation
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()

            # get lists of odd and even indices
            input_shape = input.shape[0]
            even_indices = [i for i in range(0, input_shape, 2)]
            odd_indices = [i for i in range(1, input_shape, 2)]

            # set grad_input for even_indices
            grad_input[even_indices] = (input[even_indices] >= 0).float() * grad_input[
                even_indices
            ]

            # set grad_input for odd_indices
            grad_input[odd_indices] = (input[odd_indices] < 0).float() * grad_input[
                odd_indices
            ]

        return grad_input
