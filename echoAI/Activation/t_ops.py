import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter

# Mish


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
        return input * (torch.tanh(F.softplus(input)))


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
        super(BReLU, self).__init__()

    def forward(self, input):
        """
        Forward pass of the function
        """
        return brelu_function.apply(input)


# APL



class APL(nn.Module):

    def __init__(self, s=1):
        """
        Init method.
        """
        super(APL, self).__init__()

        self.a = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.2)) for _ in range(s)])
        self.b = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.5)) for _ in range(s)])
        self.s = s

    def forward(self, input):
        """
        Forward pass of the function.
        """
        part_1 = torch.clamp_min(input, min=0.0)
        part_2 = 0
        for i in range(self.s):
            part_2 += self.a[i] * torch.clamp_min(-input+self.b[i], min=0)

        return part_1 + part_2



# Swish/ SILU/ E-Swish/ Flatten T-Swish/ Parametric Flatten T-Swish 


def swish_function(input, swish, eswish, beta, param):
    if swish is False and eswish is False:
        return input * torch.sigmoid(input)
    if swish is True and eswish is False:
        return input * torch.sigmoid(param * input)
    if eswish is True and swish is False:
        return beta * input * torch.sigmoid(input)


class Swish(nn.Module):
    def __init__(self, eswish=False, swish=False, beta=1.735, flatten=False, pfts=False):
        """
        Init method.
        """
        super(Swish, self).__init__()
        self.swish = swish
        self.eswish = eswish
        self.flatten = flatten
        self.beta = None
        self.param = None
        if eswish is not False:
            self.beta = beta
        if swish is not False:
            self.param = nn.Parameter(torch.randn(1))
            self.param.requires_grad = True
        if flatten is not False:
            if pfts is not False:
                self.const = nn.Parameter(torch.tensor(-0.2))
                self.const.requires_grad = True
            else:
                self.const = -0.2
        if eswish is not False and swish is not False and flatten is not False:
            raise RuntimeError(
                "Advisable to run either Swish or E-Swish or Flatten T-Swish"
            )

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.swish is False and self.eswish is False and self.flatten is False:
            return swish_function(input, self.swish, self.eswish, self.beta, self.param)
        if self.swish is not False:
            return swish_function(input, self.swish, self.eswish, self.beta, self.param)
        if self.eswish is not False:
            return swish_function(input, self.swish, self.eswish, self.beta, self.param)
        if self.flatten is not False:
            return (input >= 0).float() * ((input * swish_function(input, self.swish, self.eswish, self.beta, self.param)) + self.const) + (input < 0).float() * self.const



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
            return (input >= 0).float() * swish_function(
                input, False, False, None, None
            ) + (input < 0).float() * (torch.exp(input) - 1) * torch.sigmoid(input)
        else:
            return (input >= 0).float() * input * torch.max(
                self.a, torch.min(self.b, (input + 1.0) / 2.0)
            ) + (input < 0).float() * (
                torch.exp(input - 1)
                * torch.max(self.a, torch.min(self.b, (input + 1.0) / 2.0))
            )


# ISRU/ ISRLU


def isru(input, alpha):
    return input / (torch.sqrt(1 + alpha * torch.pow(input, 2)))


class ISRU(nn.Module):
    def __init__(self, alpha=1.0, isrlu=False):
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
            return (input < 0).float() * isru(input, self.apha) + (
                input >= 0
            ).float() * input
        else:
            return isru(input, self.apha)


# Maxout


class Maxout(nn.Module):
    def __init__(self, pool_size=1):
        """
        Init method.
        """
        super(Maxout, self).__init__()
        self._pool_size = pool_size

    def forward(self, input):
        """
        Forward pass of the function.
        """
        assert input.shape[1] % self._pool_size == 0, \
            'Wrong input last dim size ({}) for Maxout({})'.format(
                input.shape[1], self._pool_size)
        m, i = input.view(*input.shape[:1], input.shape[1] // self._pool_size,
                      self._pool_size, *input.shape[2:]).max(2)
        return m


# NLReLU


class NLReLU(nn.Module):
    def __init__(self, beta=1.0, inplace=False):
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
        return (1 / self.alpha) * torch.log(
            (1 + torch.exp(self.alpha * input))
            / (1 + torch.exp(self.alpha * (input - 1)))
        )


# Soft Exponential


class SoftExponential(nn.Module):
    def __init__(self, alpha=None):
        """
        Init method.
        """
        super(SoftExponential, self).__init__()

        # initialize alpha
        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))

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
        super(SQNL, self).__init__()

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

    def forward(self, input):
        """
        Forward pass of the function
        """
        return (
            (input >= self.tr).float() * (self.tr + self.ar * (input + self.tr))
            + (input < self.tr).float() * (input > self.tl).float() * input
            + (input <= self.tl).float() * (self.tl + self.al * (input + self.tl))
        )


# Funnel Activation


class Funnel(nn.Module):
    def __init__(self, in_channels):
        """
        Init method.
        """
        super(Funnel, self).__init__()
        self.conv_funnel = nn.Conv2d(
            in_channels, in_channels, 3, 1, 1, groups=in_channels
        )
        self.bn_funnel = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        """
        Forward pass of the function
        """
        tau = self.conv_funnel(input)
        tau = self.bn_funnel(tau)
        output = torch.max(input, tau)
        return output


# SLAF


class SLAF(nn.Module):
    def __init__(self, k=2):
        """
        Init method.
        """
        super(SLAF, self).__init__()
        self.k = k
        self.coeff = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.0)) for i in range(k)])

    def forward(self, input):
        """
        Forward pass of the function
        """
        out = sum([self.coeff[k] * torch.pow(input, k) for k in range(self.k)])
        return out


# AReLU


class AReLU(nn.Module):
    def __init__(self, alpha=0.90, beta=2.0):
        super(AReLU, self).__init__()
        """
        Init method.
        """
        self.alpha = nn.Parameter(torch.tensor([alpha]))
        self.beta = nn.Parameter(torch.tensor([beta]))

    def forward(self, input):
        """
        Forward pass of the function
        """
        alpha = torch.clamp(self.alpha, min=0.01, max=0.99)
        beta = 1 + torch.sigmoid(self.beta)

        return F.relu(input) * beta - F.relu(-input) * alpha



# FReLU


class FReLU(nn.Module):
    def __init__(self, in_channels):
        """
        Init method.
        """
        super(FReLU, self).__init__()
        self.bias = nn.Parameter(torch.randn(1))
        self.bias.requires_grad = True

    def forward(self, input):
        """
        Forward pass of the function
        """
        return F.relu(input) + self.bias
