# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""

# Import Necessary Modules
import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class WeightedTanh(Layer):
    '''
    Weighted TanH Activation Function.

    .. math::

        Weighted TanH(x, weight) = tanh(x * weight)

    Plot:

    .. figure::  _static/weighted_tanh.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - weight: hyperparameter (default=1.0)
    '''

    def __init__(self, input_weight):
        super(WeightedTanh, self).__init__()
        self.input_weight = input_weight

    def call(self, inputs):
        return tf.math.tanh(self.input_weight * inputs)


class Swish(Layer):
    '''
    Swish Activation Function.

    .. math::

        Swish(x, \\beta) = x*sigmoid(\\beta*x) = \\frac{x}{(1+e^{-\\beta*x})}

    Plot:

    .. figure::  _static/swish.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        -  a constant or a trainable parameter (default=1; which is equivalent to  Sigmoid-weighted Linear Unit (SiL))

    References:

        - See Swish paper:
        https://arxiv.org/pdf/1710.05941.pdf
    '''

    def __init__(self, beta):
        super(Swish, self).__init__()
        self.beta = beta

    def call(self, inputs):
        return inputs * tf.math.sigmoid(self.beta * inputs)


class ESwish(Layer):
    '''
    E-Swish Activation Function.

    .. math::
        ESwish(x, \\beta) = \\beta*x*sigmoid(x)

    Plot:

    .. figure::  _static/eswish.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - beta: a constant parameter (default value = 1.375)

    References:

        - See related paper:
        https://arxiv.org/abs/1801.07145
    '''

    def __init__(self, beta):
        super(ESwish, self).__init__()
        self.beta = beta

    def call(self, inputs):
        return self.beta * inputs * tf.math.sigmoid(inputs)


class Aria2(Layer):
    '''
    Aria-2 Activation Function.

    .. math::
        Aria2(x, \\alpha, \\beta) = (1+e^{-\\beta*x})^{-\\alpha}

    Plot:

    .. figure::  _static/aria2.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - alpha: hyper-parameter which has a two-fold effect; it reduces the curvature in 3rd quadrant as well as increases the curvature in first quadrant while lowering the value of activation (default = 1)

        - beta: the exponential growth rate (default = 0.5)

    References:

        - See Aria paper:
            https://arxiv.org/abs/1805.08878
    '''

    def __init__(self, alpha, beta):
        super(Aria2, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def call(self, inputs):
        return tf.math.pow(1 + tf.math.exp(-self.beta * inputs), -self.alpha)


class Mila(Layer):
    '''
    Mila Activation Function.

    .. math::
        Mila(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x)

    Plot:

    .. figure::  _static/mila.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - beta: scale to control the concavity of the global minima of the function (default = -0.25)

    References:

        -  https://github.com/digantamisra98/Mila
    '''

    def __init__(self, beta):
        super(Mila, self).__init__()
        self.beta = beta

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.log(1 + tf.math.exp(self.beta + inputs)))


class ISRU(Layer):
    '''
    ISRU (Inverse Square Root Unit) Activation Function.

    .. math::
        ISRU(x) = \\frac{x}{\\sqrt{1 + \\alpha * x^2}}

    Plot:

    .. figure::  _static/isru.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - alpha: A constant (default = 1.0)

    References:

        - ISRU paper:
        https://arxiv.org/pdf/1710.09967.pdf
    '''

    def __init__(self, alpha):
        super(ISRU, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        return inputs / tf.math.sqrt(1 + self.alpha * tf.math.pow(inputs, 2))


class BentIdentity(Layer):
    '''
    Bent's Identity Activation Function.

    .. math::
        bentId(x) = x + \\frac{\\sqrt{x^{2}+1}-1}{2}

    Plot:

    .. figure::  _static/bent_id.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.
    '''

    def __init__(self):
        super(BentIdentity, self).__init__()

    def call(self, inputs):
        return inputs + (tf.math.sqrt(tf.math.pow(inputs, 2) + 1) - 1) / 2


class SoftClipping(Layer):
    '''
    Soft Clipping Activation Function.

    .. math::
        SC(x) = 1 / \\alpha * log(\\frac{1 + e^{\\alpha * x}}{1 + e^{\\alpha * (x-1)}})

    Plot:

    .. figure::  _static/sc.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - alpha: hyper-parameter, which determines how close to linear the central region is and how sharply the linear region turns to the asymptotic values

    References:

        - See SC paper:
            https://arxiv.org/pdf/1810.11509.pdf
    '''

    def __init__(self, alpha):
        super(SoftClipping, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        return tf.math.log((1 + tf.math.exp(self.alpha * inputs)) * tf.math.sigmoid(self.alpha * (1 - inputs)))


class Mish(Layer):
    '''
    Mish Activation Function.

    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Plot:

    .. figure::  _static/mish.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:

        - See Mish Repository:
            https://github.com/digantamisra98/Mish
    '''

    def __init__(self):
        super(Mish, self).__init__()

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.softplus(inputs))


class BetaMish(Layer):
    '''
    β mish activation function.

    .. math::
        \\beta mish(x) = x * tanh(ln((1 + e^{x})^{\\beta}))

    Plot:

    .. figure::  _static/beta_mish.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - beta: A constant or a trainable parameter (default = 1.5)

    References

        - β-Mish: An uni-parametric adaptive activation function derived from Mish:
        https://github.com/digantamisra98/Beta-Mish)
    '''

    def __init__(self, beta):
        super(BetaMish, self).__init__()
        self.beta = beta

    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.log(tf.math.pow(1 + tf.math.exp(inputs), self.beta)))


class ELiSH(Layer):
    '''
    ELiSH (Exponential Linear Sigmoid SquasHing) Activation Function.

    .. math::
        ELiSH(x) = \\left\\{\\begin{matrix} x / (1+e^{-x}), x \\geq 0 \\\\ (e^{x} - 1) / (1 + e^{-x}), x < 0 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/elish.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:

        - Related paper:
        https://arxiv.org/pdf/1710.05941.pdf
    '''

    def __init__(self):
        super(ELiSH, self).__init__()

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs * tf.math.sigmoid(inputs)
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * (tf.math.exp(inputs) - 1) * tf.math.sigmoid(inputs)
        return case_1 + case_2


class HardELiSH(Layer):
    '''
    Hard ELiSH Activation Function.

    .. math::
        HardELiSH(x) = \\left\\{\\begin{matrix} x \\times max(0, min(1, (x + 1) / 2)), x \\geq 0 \\\\ (e^{x} - 1)\\times max(0, min(1, (x + 1) / 2)), x < 0 \\end{matrix}\\right.

    Plot:
    .. figure::  _static/hard_elish.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:

        - Related paper:
        https://arxiv.org/pdf/1710.05941.pdf
    '''

    def __init__(self):
        super(HardELiSH, self).__init__()

    def call(self, inputs):
        common = tf.math.maximum(tf.cast(0, 'float32'), tf.math.minimum(tf.cast(1, 'float32'), (inputs + 1) / 2))
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs * common
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * (tf.math.exp(inputs) - 1) * common
        return case_1 + case_2


class SineReLU(Layer):
    '''
    Sine ReLU Activation Function.

    .. math::
        SineReLU(x, \\epsilon) = \\left\\{\\begin{matrix} x , x > 0 \\\\ \\epsilon * (sin(x)-cos(x)), x \\leq 0 \\end{matrix}\\right.

    Plot:
    .. figure::  _static/sine_relu.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:

        - See related Medium article:
        https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d

    Arguments:

        - epsilon: hyperparameter (default=0.01)
    '''

    def __init__(self, epsilon):
        super(SineReLU, self).__init__()
        self.epsilon = epsilon

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, 0), 'float32') * inputs
        case_2 = tf.cast(tf.math.less_equal(self.epsilon * (tf.math.sin(inputs) - tf.math.cos(inputs)), 0), 'float32')
        return case_1 + case_2


class FlattenTSwish(Layer):
    '''
    FTS (Flatten T-Swish) Activation Function.

    .. math::
        FTS(x) = \\left\\{\\begin{matrix} \\frac{x}{1 + e^{-x}} , x \\geq  0 \\\\ 0, x < 0 \\end{matrix}\\right.

    Plot:
    .. figure::  _static/fts.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:

        - Flatten T-Swish paper:
        https://arxiv.org/pdf/1812.06247.pdf
    '''

    def __init__(self):
        super(FlattenTSwish, self).__init__()

    def call(self, inputs):
        return tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs * tf.math.sigmoid(inputs)


class SQNL(Layer):
    '''
    SQNL Activation Function.

    .. math::
        SQNL(x) = \\left\\{\\begin{matrix} 1, x > 2 \\\\ x - \\frac{x^2}{4}, 0 \\leq x \\leq 2 \\\\  x + \\frac{x^2}{4}, -2 \\leq x < 0 \\\\ -1, x < -2 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/sqnl.png
        :align:   center

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:

        - SQNL Paper:
        https://ieeexplore.ieee.org/document/8489043
    '''

    def __init__(self):
        super(SQNL, self).__init__()

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, 2), 'float32')
        case_2 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * tf.cast(tf.math.less_equal(inputs, 2), 'float32') * (inputs - tf.math.pow(inputs, 2) / tf.cast(4, 'float32'))
        case_3 = tf.cast(tf.math.greater_equal(inputs, -2), 'float32') * tf.cast(tf.math.less(inputs, 0), 'float32') * (inputs + tf.math.pow(inputs, 2) / tf.cast(4, 'float32'))
        case_4 = tf.cast(tf.math.less(inputs, -2), 'float32') * tf.cast(-1, 'float32')
        return case_1 + case_2 + case_3 + case_4


class ISRLU(Layer):
    '''
    ISRLU Activation Function.

    .. math::
        ISRLU(x)=\\left\\{\\begin{matrix} x, x\\geq 0 \\\\  x * (\\frac{1}{\\sqrt{1 + \\alpha*x^2}}), x <0 \\end{matrix}\\right.

    Plot:

    .. figure::  _static/isrlu.png
        :align:   center

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - alpha: hyperparameter α controls the value to which an ISRLU saturates for negative inputs (default = 1)

    References:

        - ISRLU paper: https://arxiv.org/pdf/1710.09967.pdf
    '''

    def __init__(self, alpha):
        super(ISRLU, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * inputs  / tf.math.sqrt(1 + self.alpha * tf.math.pow(inputs, 2))
        return case_1 + case_2


class SoftExponential(Layer):
    '''
    Soft-Exponential Activation Function with 1 trainable parameter..

    .. math::
        SoftExponential(x, \\alpha) = \\left\\{\\begin{matrix} - \\frac{log(1 - \\alpha(x + \\alpha))}{\\alpha}, \\alpha < 0\\\\  x, \\alpha = 0\\\\  \\frac{e^{\\alpha * x} - 1}{\\alpha} + \\alpha, \\alpha > 0 \\end{matrix}\\right.

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Parameters:

        - alpha - trainable parameter

    References:

        - See Soft-Exponential paper:
        https://arxiv.org/pdf/1602.01321.pdf
    '''

    def __init__(self, alpha):
        super(SoftExponential, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        condition_1 = tf.cast(tf.math.less(self.alpha, 0), 'float32')
        condition_2 = tf.cast(tf.math.equal(self.alpha, 0), 'float32')
        condition_3 = tf.cast(tf.math.greater(self.alpha, 0), 'float32')
        case_1 = condition_1 * (-1 / self.alpha) * tf.math.log(1 - self.alpha * (inputs + self.alpha))
        case_2 = condition_2 * inputs
        case_3 = condition_3 * (self.alpha + (1 / self.alpha) * (tf.math.exp(self.alpha * inputs) - 1))
        return case_1 + case_2 + case_3


class CELU(Layer):
    '''
    CELU Activation Function.

    .. math::
        CELU(x, \\alpha) = max(0,x) + min(0,\\alpha * (exp(x/ \\alpha)-1))

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - alpha: the α value for the CELU formulation (default=1.0)

    References:

        - See CELU paper:
            https://arxiv.org/abs/1704.07483
    '''

    def __init__(self, alpha):
        super(CELU, self).__init__()
        self.alpha = alpha

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * self.alpha * (tf.math.exp(inputs / self.alpha) - 1)
        return case_1 + case_2


class HardTanh(Layer):
    '''
    Hard-TanH Activation Function.

    .. math::
        Hard-TanH(x) = \\left\\{\\begin{matrix} 1, x > 1 \\\\   x , -1 \\leq x \\leq 1 \\\\ -1, x <- 1 \\end{matrix}\\right.

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.
    '''

    def __init__(self):
        super(HardTanh, self).__init__()

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, 1), 'float32')
        case_2 = tf.cast(tf.math.less(inputs, -1), 'float32') * -1
        case_3 = tf.cast(tf.math.greater_equal(inputs, -1), 'float32') * tf.cast(tf.math.less_equal(inputs, 1), 'float32') * inputs
        return case_1 + case_2 + case_3


class LogSigmoid(Layer):
    '''
    Log-Sigmoid Activation Function.

    .. math::
        Log-Sigmoid(x) = log (\\frac{1}{1+e^{-x}})

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.
    '''

    def __init__(self):
        super(LogSigmoid, self).__init__()

    def call(self, inputs):
        return tf.math.log(tf.math.sigmoid(inputs))


class TanhShrink(Layer):
    '''
    TanH-Shrink Activation Function.

    .. math::
        TanH-Shrink(x) = x - tanh(x)

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.
    '''

    def __init__(self):
        super(TanhShrink, self).__init__()

    def call(self, inputs):
        return inputs - tf.math.tanh(inputs)


class HardShrink(Layer):
    '''
    Hard-Shrink Activation Function.

    .. math::
        Hard-Shrink(x) = \\left\\{\\begin{matrix} x, x > \\lambda \\\\   0 , - \\lambda \\leq x \\leq \\lambda \\\\ x, x <- \\lambda \\end{matrix}\\right.

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - lambda: the λ value for the Hardshrink formulation (default=0.5)
    '''

    def __init__(self, _lambda = 0.5):
        super(HardShrink, self).__init__()
        self._lambda = _lambda

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, self._lambda), 'float32') * inputs
        case_2 = tf.cast(tf.math.less(inputs, -1 * self._lambda), 'float32') * inputs
        return case_1 + case_2


class SoftShrink(Layer):
    '''
    Soft-Shrink Activation Function.

    .. math::
        Soft-Shrink(x) = \\left\\{\\begin{matrix} x - \\lambda , x > \\lambda \\\\   0 , - \\lambda \\leq x \\leq \\lambda \\\\ x + \\lambda , x <- \\lambda \\end{matrix}\\right.

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Arguments:

        - lambda: the λ value for the Softshrink formulation (default=0.5)
    '''

    def __init__(self, _lambda = 0.5):
        super(HardShrink, self).__init__()
        self._lambda = _lambda

    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, self._lambda), 'float32') * (inputs - self._lambda)
        case_2 = tf.cast(tf.math.less(inputs, -1 * self._lambda), 'float32') * (inputs - self._lambda)
        return case_1 + case_2


class SoftMin(Layer):
    '''
    SoftMin Activation Function.

    .. math::
        SoftMin(x) = Softmax(-x)

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.
    '''

    def __init__(self):
        super(SoftMin, self).__init__()

    def call(self, inputs):
        return tf.math.softmax(-inputs)


class LogSoftMax(Layer):
    '''
    Log-SoftMax Activation Function.

    .. math::
        Log-SoftMax(x) = log(Softmax(-x))

    Shape:

        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.
    '''

    def __init__(self):
        super(LogSoftMax, self).__init__()

    def call(self, inputs):
        return tf.math.log(tf.math.softmax(inputs))


class MaxOut(Layer):
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
    '''

    def __init__(self):
        super(MaxOut, self).__init__()

    def call(self, inputs):
        return K.max(inputs)


class SReLU(Layer):
    '''
    SReLU (S-shaped Rectified Linear Activation Unit): a combination of three linear functions, which perform mapping R → R with the following formulation:

    .. math::
        h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.

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
    '''

    def __init__(self, t, a, r, l):
        super(SReLU, self).__init__()
        self.t = tf.cast(t, 'float32')
        self.a = tf.cast(a, 'float32')
        self.r = tf.cast(r, 'float32')
        self.l = tf.cast(l, 'float32')

    def call(self, inputs):
        condition_1 = tf.cast(tf.math.greater_equal(inputs, tf.math.pow(self.t, self.r)), 'float32')
        condition_2 = tf.cast(tf.math.greater(tf.math.pow(self.t, self.r), inputs), 'float32') + tf.cast(tf.math.greater(inputs, tf.math.pow(self.t, self.l)), 'float32')
        condition_3 = tf.cast(tf.math.less_equal(inputs, tf.math.pow(self.t, self.l)), 'float32')
        case_1 = condition_1 * (tf.math.pow(self.t, self.r) + tf.math.pow(self.a, self.r) * (inputs - tf.math.pow(self.t, self.r)))
        case_2 = condition_2 * inputs
        case_3 = condition_3 * (tf.math.pow(self.t, self.l) + tf.math.pow(self.a, self.l) * (inputs - tf.math.pow(self.t, self.l)))
        return case_1 + case_2 + case_3


class BReLU(Layer):
    '''
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
    '''

    def __init__(self):
        super(BReLU, self).__init__()

    def call(self, inputs):
        def brelu(x):
            #get shape of X, we are interested in the last axis, which is constant
            shape = K.int_shape(x)
            #last axis
            dim = shape[-1]
            #half of the last axis (+1 if necessary)
            dim2 = dim // 2
            if dim % 2 != 0:
                dim2 += 1
            #multiplier will be a tensor of alternated +1 and -1
            multiplier = K.ones((dim2,))
            multiplier = K.stack([multiplier, -multiplier], axis = -1)
            if dim % 2 != 0:
                multiplier = multiplier[:-1]
            #adjust multiplier shape to the shape of x
            multiplier = K.reshape(multiplier, tuple(1 for _ in shape[:-1]) + (-1,))
            return multiplier * tf.nn.relu(multiplier * x)
        return Lambda(brelu)(inputs)


class APL(Layer):
    '''
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
    '''

    def __init__(
        self,
        alpha_initializer = 'zeros',
        b_initializer = 'zeros',
        S = 1,
        alpha_regularizer = None,
        b_regularizer = None,
        alpha_constraint = None,
        b_constraint = None,
        shared_axes = None,
        **kwargs):
        super(APL, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.b_initializer = initializers.get(b_initializer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.b_constraint = constraints.get(b_constraint)
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)
        self.S = S
        self.alpha_arr=[]
        self.b_arr=[]

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.param_broadcast = [False] * len(param_shape)
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
                self.param_broadcast[i - 1] = True
        for i in range(self.S):
            self.alpha_arr.append(
                self.add_weight(
                    shape = param_shape,
                    name = 'alpha_' + str(i),
                    initializer = self.alpha_initializer,
                    regularizer = self.alpha_regularizer,
                    constraint = self.alpha_constraint
                )
            )
            self.b_arr.append(
                self.add_weight(
                    shape = param_shape,
                    name = 'b_' + str(i),
                    initializer = self.b_initializer,
                    regularizer = self.b_regularizer,
                    constraint = self.b_constraint
                )
            )
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
        self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs, mask = None):
        max_a = tf.maximum(0., inputs)
        max_b = 0
        for i in range(self.S):
            max_b += self.alpha_arr[i] * tf.maximum(0., -inputs + self.b_arr[i])
        return max_a + max_b

    def get_config(self):
        config = {
            'alpha_initializer': initializers.serialize(self.b_initializer),
            'alpha_regularizer': regularizers.serialize(self.b_regularizer),
            'alpha_constraint': constraints.serialize(self.b_constraint),
            'b_initializer': initializers.serialize(self.b_initializer),
            'b_regularizer': regularizers.serialize(self.b_regularizer),
            'b_constraint': constraints.serialize(self.b_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(APL, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
