'''
Script implements Maxout activation

.. math::

    maxout(\\vec{x}) = max_i(x_i)

See related paper:
https://arxiv.org/pdf/1302.4389.pdf

See implementation:
https://github.com/Usama113/Maxout-PyTorch/blob/master/Maxout.ipynb
'''

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.autograd import Function

class Maxout(Function):
    '''
    Implementation of Maxout:

        .. math::

            maxout(\\vec{x}) = max_i(x_i)

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        - See Maxout paper:
        https://arxiv.org/pdf/1302.4389.pdf

        - Reference to the implementation:
        https://github.com/Usama113/Maxout-PyTorch/blob/master/Maxout.ipynb

    Examples:
        >>> a1 = Maxout.apply
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    @staticmethod
    def forward(ctx, input):
        x = input
        kernels = x.shape[1]  # to get how many kernels/output
        max_out=4    #Maxout Parameter
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x = x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        ctx.max_out=max_out
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input1,indices,max_out= ctx.saved_variables[0],Variable(ctx.indices),ctx.max_out
        input=input1.clone()
        for i in range(max_out):
            a0=indices==i
            input[:,i:input.data.shape[1]:max_out]=a0.float()*grad_output

        return input
