import megengine.functional as F
import megengine.module as M
import megengine as mge
import numpy as np


# FReLU


class FReLU(M.Module):

    def __init__(self, in_channels):
        """
        Init method.
        """
        super(FReLU,self).__init__()
        self.conv_frelu = M.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.bn_frelu = M.BatchNorm2d(in_channels)

    def forward(self, input):
        """
        Forward pass of the function.
        """
        tau = self.conv_frelu(input)
        tau = self.bn_frelu(tau)
        output = F.maximum(input, tau)
        return output


# Mish


class Mish(M.Module):

    def __init__(self):
        """
        Init method.
        """
        super(Mish,self).__init__()

    def forward(self, input):
        """
        Forward pass of the function.
        """
        return input * F.tanh(F.softplus(input))


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
        return F.pow((1 + F.exp(-self.beta * input)), -self.alpha)



# Swish/ SILU/ E-Swish/ Flatten T-Swish


def swish_function(input, swish, eswish, beta, param):
    if swish is False and eswish is False:
        return input * F.sigmoid(input)
    if swish:
        return input * F.sigmoid(param * input)
    if eswish:
        return beta * input * F.sigmoid(input)


class Swish(nn.Module):

    def __init__(self, eswish=False, swish=False, beta = 1.735, flatten = False):
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
            self.param = mge.Parameter(mge.tensor(np.random.randn(1)))
            self.param.requires_grad = True
        if eswish is not False and swish is not False and flatten is not False:
            raise RuntimeError('Advisable to run either Swish or E-Swish or Flatten T-Swish')

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.swish is False and self.eswish is False and self.flatten is False:
            return swish(input, self.swish, self.eswish, self.beta, self.param)
        if self.swish is not False:
            return swish(input, self.swish, self.eswish, self.beta, self.param)
        if self.eswish is not False:
            return swish(input, self.swish, self.eswish, self.beta, self.param)
        if self.flatten is not False:
            return F.clip(input * F.sigmoid(input), lower=0)


# ELisH/ Hard-ELisH


class Elish(nn.Module):

    def __init__(self, hard=False):
        """
        Init method.
        """
        super(Elish, self).__init__()
        self.hard = hard
        if hard is not False:
            self.a = mge.tensor(0.0)
            self.b = mge.tensor(1.0)

    def forward(self, input):
        """
        Forward pass of the function.
        """
        if self.hard is False:
            return (input >= 0).float() * swish(input, False, False, None, None) + (input < 0).float() * (F.exp(input) - 1) * F.sigmoid(input)
        else:
            return (input >= 0).float() * input * F.max(self.a,F.min(self.b, (input + 1.0) / 2.0)) + (input < 0).float() * (F.exp(input - 1) * F.max(self.a, F.min(self.b, (input + 1.0) / 2.0)))


# ISRU/ ISRLU

def isru(input, alpha):
    return input / (F.sqrt(1 + alpha * F.pow(input, 2)))


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
            return (input < 0).float() * isru(input, self.apha) + (input >= 0).float() * input
        else:
            return isru(input, self.apha)


# NLReLU


class NLReLU(nn.Module):

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
        return F.log(1 + self.beta * F.relu(x))



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
        return (1 / self.alpha) * F.log((1 + F.exp(self.alpha * input)) / (1 + F.exp(self.alpha * (input - 1))))



# Soft Exponential


class SoftExponential(nn.Module):

    def __init__(self, alpha=None):
        """
        Init method.
        """
        super(SoftExponential, self).__init__()

        # initialize alpha
        if alpha is None:
            self.alpha = mge.Parameter(mge.tensor(0.0))
        else:
            self.alpha = mge.Parameter(mge.tensor(alpha))
        self.alpha.requiresGrad = True

    def forward(self, x):
        """
        Forward pass of the function
        """
        if self.alpha == 0.0:
            return x

        if self.alpha < 0.0:
            return -F.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        if self.alpha > 0.0:
            return (F.exp(self.alpha * x) - 1) / self.alpha + self.alpha



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
        + (input - F.pow(input, 2) / 4)
        * (input >= 0).float()
        * (input <= 2).float()
        + (input + F.pow(input, 2) / 4)
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
            self.tr = mge.Parameter(mge.tensor(np.random.randn(in_features).astype(np.float32)))
            self.tl = mge.Parameter(mge.tensor(np.random.randn(in_features).astype(np.float32)))
            self.ar = mge.Parameter(mge.tensor(np.random.randn(in_features).astype(np.float32)))
            self.al = mge.Parameter(mge.tensor(np.random.randn(in_features).astype(np.float32)))

            self.tr.requiresGrad = True
            self.tl.requiresGrad = True
            self.ar.requiresGrad = True
            self.al.requiresGrad = True

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