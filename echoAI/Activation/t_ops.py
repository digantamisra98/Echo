import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F


# Mish


class mish_function(Function):
    if torch.cuda.is_available(): 
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            y = x * torch.tanh(F.softplus(x))
            return y

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            sigmoid = torch.sigmoid(x)
            tanh_sp = torch.tanh(F.softplus(x)) 
            return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))
    else:
        @torch.jit.script
        def mish(input):
            delta = torch.exp(-input)
            alpha = 1 + 2 * delta
            return input * alpha / (alpha + 2* delta * delta)


class Mish(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(Mish, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return mish_function.apply(input)


# Aria 2


class Aria2(nn.Module):

    def __init__(self, beta=0.5, alpha=1.0):
        """
        Init method.
        """
        super(Aria2, self).__init__()
        self.beta = beta
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return torch.pow((1 + torch.exp(-self.beta * input)), -self.alpha)


# BReLU


class brelu_function(Function):

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


class BReLU(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(BreLU, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function
        """
        return brelu_function.apply(input)


# APL


class apl_function(Function):

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

    def __init__(self, in_features, a=None, b=None):
        """
        Init method.
        """
        super(APL, self).__init__()
        self.in_features = in_features

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

    def forward(self, input):
        """
        Forward pass of the function
        """
        return apl_function.apply(input, a, b)


# ELisH/ Hard-ELisH


class Elish(nn.Module):

    def __init__(self, hard=False):
        """
        Init method.
        """
        super(Elish, self).__init__()
        self.hard = hard
        if hard is not False:
            self.a = torch.tensor(0.0)
            self.b = torch.tensor(1.0)

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.hard is False:
            return (input >= 0).float() * input * torch.sigmoid(input) + (input < 0).float() * (torch.exp(input) - 1) * torch.sigmoid(input)
        else:
            return (input >= 0).float() * input * torch.max(self.a,torch.min(self.b, (input + 1.0) / 2.0)) + (input < 0).float() * (torch.exp(input - 1) * torch.max(self.a, torch.min(self.b, (input + 1.0) / 2.0)))


# Swish/ SILU/ E-Swish/ Flatten T-Swish


class Swish(nn.Module):

    def __init__(self, eswish=False, swish=False, beta = 1.735, flatten = False):
        """
        Init method.
        """
        super(Swish, self).__init__()
        self.swish = swish
        self.eswish = eswish
        self.flatten = flatten
        if eswish is not False:
            self.beta = beta
        if swish is not False:
            self.param = nn.Parameter(torch.randn(1))
            self.param.requires_grad = True
        if eswish is not False and swish is not False and flatten is not False:
            raise RuntimeError('Advisable to run either Swish or E-Swish or Flatten T-Swish')

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.swish is False and self.eswish is False and self.flatten is False:
            return input * torch.sigmoid(input)
        if self.swish is not False:
            return input * torch.sigmoid(self.param * input)
        if self.eswish is not False:
            return self.beta * input * torch.sigmoid(input)
        if self.flatten is not False:
            return torch.clamp(input * torch.sigmoid(input), min=0)


# ISRU/ ISRLU


class ISRU(nn.Module):

    def __init__(self, alpha=1.0, isrlu = False):
        """
        Init method.
        """
        super().__init__()
        self.alpha = alpha
        self.isrlu = isrlu

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.isrlu is not False:
            return (input < 0).float() * input / (torch.sqrt(1 + self.alpha * torch.pow(input, 2))) + (input >= 0).float() * input
        else:
            return input / (torch.sqrt(1 + self.alpha * torch.pow(input, 2)))



# Maxout


class maxout_function(Function):

    @staticmethod
    def forward(ctx, input):
        x = input
        kernels = x.shape[1]  # to get how many kernels/output
        max_out = 4  # Maxout Parameter
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x = x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices = indices
        ctx.max_out = max_out
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input1, indices, max_out = (
            ctx.saved_variables[0],
            Variable(ctx.indices),
            ctx.max_out,
        )
        input = input1.clone()
        for i in range(max_out):
            a0 = indices == i
            input[:, i : input.data.shape[1] : max_out] = a0.float() * grad_output

        return input
        

class Maxout(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(Maxout, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function
        """
        return maxout_function.apply(input)



# NLReLU


class NLReLU(nn.Module):

    def __init__(self, beta=1.0, inplace = False):
        """
        Init method.
        """
        super().__init__()
        self.beta = beta
        self.inplace = inplace

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if inplace:
            return torch.log(F.relu_(x).mul_(self.beta).add_(1), out=x)
        else:
            return torch.log(1 + self.beta * F.relu(x))



# Soft Clipping


class SoftClipping(nn.Module):

    def __init__(self, alpha=0.5):
        """
        Init method.
        """
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return (1 / self.alpha) * torch.log((1 + torch.exp(self.alpha * input)) / (1 + torch.exp(self.alpha * (input - 1))))



# Soft Exponential


class SoftExponential(nn.Module):

    def __init__(self, in_features, alpha=None):
        """
        Init method.
        """
        super(SoftExponential, self).__init__()
        self.in_features = in_features

        # initialize alpha
        if alpha is None:
            self.alpha = Parameter(torch.tensor(0.0))
        else:
            self.alpha = Parameter(torch.tensor(alpha))

        self.alpha.requiresGrad = True

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



# SQNL


class SQNL(nn.Module):

    def __init__(self):
        """
        Init method.
        """
        super(SQNL,self).__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return (
        (input > 2).float()
        + (input - torch.pow(input, 2) / 4)
        * (input >= 0).float()
        * (input <= 2).float()
        + (input + torch.pow(input, 2) / 4)
        * (input < 0).float()
        * (input >= -2).float()
        - (input < -2).float()
        )



# SReLU


class SReLU(nn.Module):

    def __init__(self, in_features, parameters=None):
        """
        Init method.
        """
        super(SReLU, self).__init__()
        self.in_features = in_features

        if parameters is None:
            self.tr = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
            self.tl = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
            self.ar = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
            self.al = Parameter(
                torch.randn(in_features, dtype=torch.float, requires_grad=True)
            )
        else:
            self.tr, self.tl, self.ar, self.al = parameters

    def forward(self, x):
        """
        Forward pass of the function
        """
        return (
            (x >= self.tr).float() * (self.tr + self.ar * (x + self.tr))
            + (x < self.tr).float() * (x > self.tl).float() * x
            + (x <= self.tl).float() * (self.tl + self.al * (x + self.tl))
        )