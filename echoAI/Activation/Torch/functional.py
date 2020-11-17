"""
Script provides functional interface for custom activation functions.
"""

# import pytorch
import torch
import torch.nn.functional as F


def silu(input, inplace=False):
    """
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:

    .. math::

        SiLU(x) = x * sigmoid(x)

    See additional documentation for :mod:`echoAI.Activation.Torch.silu`.
    """
    if inplace:
        result = input.clone()
        torch.sigmoid_(input)
        input *= result
    else:
        return input * torch.sigmoid(input)


def aria2(input, beta=1, alpha=1.5):
    """
    Applies the Aria-2 function element-wise:

    .. math::

        Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

    See additional documentation for :mod:`echoAI.Activation.Torch.aria2`.
    """
    return torch.pow((1 + torch.exp(-beta * input)), -alpha)


def eswish(input, beta=1.75):
    """
    Applies the E-Swish function element-wise:

        .. math::

            ESwish(x, \\beta) = \\beta*x*sigmoid(x)

    See additional documentation for :mod:`echoAI.Activation.Torch.eswish`.
    """
    return beta * input * torch.sigmoid(input)


def swish(input, beta=1.25):
    """
    Applies the Swish function element-wise:

        .. math::

            Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

    See additional documentation for :mod:`echoAI.Activation.Torch.swish`.
    """
    return input * torch.sigmoid(beta * input)


def elish(input):
    """
    Applies the ELiSH (Exponential Linear Sigmoid SquasHing) function element-wise:

    See additional documentation for :mod:`echoAI.Activation.Torch.elish`.

        .. math::

            ELiSH(x) = \\left\\{\\begin{matrix} x / (1+e^{-x}), x \\geq 0 \\\\ (e^{x} - 1) / (1 + e^{-x}), x < 0 \\end{matrix}\\right.

    See additional documentation for :mod:`echoAI.Activation.Torch.elish`.
    """
    return (input >= 0).float() * input * torch.sigmoid(input) + (input < 0).float() * (
        torch.exp(input) - 1
    ) / (torch.exp(-input) + 1)


def hard_elish(input):
    """
    Applies the HardELiSH (Exponential Linear Sigmoid SquasHing) function element-wise:

        .. math::

            HardELiSH(x) = \\left\\{\\begin{matrix} x \\times max(0, min(1, (x + 1) / 2)), x \\geq 0 \\\\ (e^{x} - 1)\\times max(0, min(1, (x + 1) / 2)), x < 0 \\end{matrix}\\right.

    See additional documentation for :mod:`echoAI.Activation.Torch.hard_elish`.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return (input >= 0).float() * input * torch.max(
        torch.tensor(0.0, device=device),
        torch.min(torch.tensor(1.0, device=device), (input + 1.0) / 2.0),
    ) + (input < 0).float() * (
        torch.exp(input - 1)
        * torch.max(
            torch.tensor(0.0, device=device),
            torch.min(torch.tensor(1.0, device=device), (input + 1.0) / 2.0),
        )
    )


def fts(input):
    """
    Applies the FTS (Flatten T-Swish) activation function element-wise:

    .. math::

        FTS(x) = \\left\\{\\begin{matrix} \\frac{x}{1 + e^{-x}} , x \\geq  0 \\\\ 0, x < 0 \\end{matrix}\\right.

    See additional documentation for :mod:`echoAI.Activation.Torch.fts`.
    """
    return torch.clamp(input / (1 + torch.exp(-input)), min=0)


def sqnl(input):
    """
    Applies the SQNL activation function element-wise:

    .. math::

        SQNL(x) = \\left\\{\\begin{matrix} 1, x > 2 \\\\ x - \\frac{x^2}{4}, 0 \\leq x \\leq 2 \\\\  x + \\frac{x^2}{4}, -2 \\leq x < 0 \\\\ -1, x < -2 \\end{matrix}\\right.

    See additional documentation for :mod:`echoAI.Activation.Torch.sqnl`.
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


def isru(input, alpha=1.0):
    """
    Applies the ISRU function element-wise:

    .. math::

        ISRU(x) = \\frac{x}{\\sqrt{1 + \\alpha * x^2}}

    See additional documentation for :mod:`echoAI.Activation.Torch.isru`.
    """
    return input / (torch.sqrt(1 + alpha * torch.pow(input, 2)))


def isrlu(input, alpha=1.0):
    """
    Applies the ISRLU function element-wise:

    .. math::

        ISRLU(x)=\\left\\{\\begin{matrix} x, x\\geq 0 \\\\  x * (\\frac{1}{\\sqrt{1 + \\alpha*x^2}}), x <0 \\end{matrix}\\right.

    See additional documentation for :mod:`echoAI.Activation.Torch.isrlu`.
    """
    return (input < 0).float() * isru(input, alpha) + (input >= 0).float() * input


def soft_clipping(input, alpha=0.5):
    """
    Applies the Soft Clipping function element-wise:

    .. math::

        SC(x) = 1 / \\alpha * log(\\frac{1 + e^{\\alpha * x}}{1 + e^{\\alpha * (x-1)}})

    See additional documentation for :mod:`echoAI.Activation.Torch.soft_clipping`.
    """
    return (1 / alpha) * torch.log(
        (1 + torch.exp(alpha * input)) / (1 + torch.exp(alpha * (input - 1)))
    )


def nl_relu(x, beta=1., inplace=False):
    """
    Applies the natural logarithm ReLU activation function element-wise:

    See additional documentation for :mod:`echoAI.Activation.Torch.nl_relu`.
    """

    if inplace:
        return torch.log(F.relu_(x).mul_(beta).add_(1), out=x)
    else:
        return torch.log(1 + beta * F.relu(x))
