# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K
from keras import initializers

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mila(beta=0.5)(X_input)

    '''

    def __init__(self, beta=-0.25, **kwargs):
        super(Mila, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs*K.tanh(K.softplus(inputs + self.beta))

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Mila, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Swish(beta=0.5)(X_input)

    '''

    def __init__(self, beta=1.0, **kwargs):
        super(Swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs*K.sigmoid(inputs*self.beta)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class Eswish(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Eswish(beta=0.5)(X_input)

    '''

    def __init__(self, beta=1.375, **kwargs):
        super(Eswish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return self.beta*inputs*K.sigmoid(inputs)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(Eswish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = BetaMish(beta=1.5)(X_input)

    '''

    def __init__(self, beta=1.5, **kwargs):
        super(BetaMish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs*K.tanh(K.log(K.pow((1+K.exp(inputs)),self.beta)))

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(BetaMish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = ISRU(alpha=0.5)(X_input)

    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(ISRU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return inputs/(K.sqrt(1 + self.alpha * K.pow(inputs,2)))

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(ISRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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
        - Mish paper:
        https://arxiv.org/abs/1908.08681

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Mish()(X_input)

    '''

    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(Mish, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SQNL()(X_input)

    '''

    def __init__(self, **kwargs):
        super(SQNL, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater(inputs , 2), 'float32')\
        + (inputs - K.pow(inputs,2)/4) * K.cast(K.greater_equal(inputs,0), 'float32') * K.cast(K.less_equal(inputs, 2), 'float32') \
        + (inputs + K.pow(inputs,2)/4) * K.cast(K.less(inputs, 0), 'float32') * K.cast(K.greater_equal(inputs, -2), 'float32') - K.cast(K.less(inputs, -2), 'float32')

    def get_config(self):
        base_config = super(SQNL, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape

class FTS(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = FTS()(X_input)

    '''

    def __init__(self, **kwargs):
        super(FTS, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs / (1 + K.exp(- inputs))

    def get_config(self):
        base_config = super(FTS, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape

class Elish(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Elish()(X_input)

    '''

    def __init__(self, **kwargs):
        super(Elish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs * K.sigmoid(inputs) + K.cast(K.less(inputs, 0), 'float32') * (K.exp(inputs) - 1) / (K.exp(- inputs) + 1)

    def get_config(self):
        base_config = super(Elish, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape

class HardElish(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = HardElish()(X_input)

    '''

    def __init__(self, **kwargs):
        super(HardElish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs * K.maximum(K.cast_to_floatx(0.0), K.minimum(K.cast_to_floatx(1.0), (inputs + 1.0)/2.0)) \
        + K.cast(K.less(inputs, 0), 'float32') * (K.exp(inputs - 1) * K.maximum(K.cast_to_floatx(0.0), K.minimum(K.cast_to_floatx(1.0), (inputs + 1.0)/2.0)))

    def get_config(self):
        base_config = super(HardElish, self).get_config()
        return dict(list(base_config.items()) 

    def compute_output_shape(self, input_shape):
        return input_shape

class BentID(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = BentID()(X_input)

    '''

    def __init__(self, **kwargs):
        super(BentID, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs + ((K.sqrt(K.pow(inputs,2)+1)-1)/2)

    def get_config(self):
        base_config = super(BentID, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = WeightedTanh(weight=1.0)(X_input)

    '''

    def __init__(self, weight=1.0, **kwargs):
        super(WeightedTanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.weight = K.cast_to_floatx(weight)

    def call(self, inputs):
        return K.tanh(inputs * self.weight)

    def get_config(self):
        config = {'weight': float(self.weight)}
        base_config = super(WeightedTanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SineReLU(epsilon=0.01)(X_input)

    '''

    def __init__(self, epsilon=0.01, **kwargs):
        super(SineReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = K.cast_to_floatx(epsilon)

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs + K.cast(K.less(inputs, 0), 'float32') * self.epsilon * (K.sin(inputs) - K.cos(inputs))

    def get_config(self):
        config = {'epsilon': float(self.epsilon)}
        base_config = super(SineReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = ISRLU(alpha=1.0)(X_input)

    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(ISRLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.cast(K.less(inputs, 0), 'float32') * ISRU(alpha=self.alpha)(inputs) + K.cast(K.greater_equal(inputs, 0), 'float32') * inputs

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(ISRLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SoftClipping(alpha=0.5)(X_input)

    '''

    def __init__(self, alpha=0.5, **kwargs):
        super(SoftClipping, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return (1 / self.alpha) * K.log((1 + K.exp(self.alpha * inputs))/(1 + K.exp(self.alpha *(inputs - 1))))

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(SoftClipping, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X =Aria2(alpha=1.0, beta=0.5)(X_input)

    '''

    def __init__(self, alpha=1.0, beta=0.5, **kwargs):
        super(Aria2, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return K.pow((1 + K.exp(-self.beta * inputs)), -self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha), 'beta': float(self.beta)}
        base_config = super(Aria2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class Celu(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = Celu(alpha=1.0)(X_input)

    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(Celu, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs + K.cast(K.less(inputs, 0), 'float32') * self.alpha * (K.exp (inputs / self.alpha) - 1)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(Celu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ReLU6(Layer):
    '''
    RELU6 Activation Function.

    .. math::

        RELU6(x) = min(max(0,x),6)

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:
        - See RELU6 paper:
            http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = ReLU6()(X_input)

    '''

    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 6), 'float32') * 6 + K.cast(K.less(inputs, 6), 'float32') * K.relu(inputs)

    def get_config(self):
        base_config = super(ReLU6, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = HardTanh()(X_input)

    '''

    def __init__(self, **kwargs):
        super(HardTanh, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater(inputs , 1), 'float32')\
        + inputs * K.cast(K.less_equal(inputs, 1), 'float32') * K.cast(K.greater_equal(inputs, -1), 'float32') - K.cast(K.less(inputs, -1), 'float32')

    def get_config(self):
        base_config = super(HardTanh, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = LogSigmoid()(X_input)

    '''

    def __init__(self, **kwargs):
        super(LogSigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.log(K.sigmoid(inputs))

    def get_config(self):
        base_config = super(LogSigmoid, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = TanhShrink()(X_input)

    '''

    def __init__(self, **kwargs):
        super(TanhShrink, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs - K.tanh(inputs)

    def get_config(self):
        base_config = super(TanhShrink, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = HardShrink(lambd = 0.5)(X_input)

    '''

    def __init__(self, lambd = 0.5, **kwargs):
        super(HardShrink, self).__init__(**kwargs)
        self.supports_masking = True
        self.lambd = K.cast_to_floatx(lambd)

    def call(self, inputs):
        return K.cast(K.greater(inputs , self.lambd), 'float32') * inputs \
        + 0 * K.cast(K.less_equal(inputs, self.lambd), 'float32') * K.cast(K.greater_equal(inputs, -self.lambd), 'float32') + inputs *  K.cast(K.less(inputs, -self.lambd), 'float32')

    def get_config(self):
        config = {'lambd': float(self.lambd)}
        base_config = super(HardShrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SoftShrink(lambd = 0.5)(X_input)

    '''

    def __init__(self, lambd= 0.5, **kwargs):
        super(SoftShrink, self).__init__(**kwargs)
        self.supports_masking = True
        self.lambd = K.cast_to_floatx(lambd)

    def call(self, inputs):
        return (K.cast(K.greater(inputs , self.lambd), 'float32') * (inputs - self.lambd)) \
        + (0 * K.cast(K.less_equal(inputs, self.lambd), 'float32') * K.cast(K.greater_equal(inputs, -self.lambd), 'float32')) \
        + ((inputs + self.lambd) *  K.cast(K.less(inputs, -self.lambd), 'float32'))

    def get_config(self):
        config = {'lambd': float(self.lambd)}
        base_config = super(SoftShrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SoftMin()(X_input)

    '''

    def __init__(self, **kwargs):
        super(SoftMin, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.softmax(-inputs)

    def get_config(self):
        base_config = super(SoftMin, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


class LogSoftmax(Layer):
    '''
    Log-SoftMax Activation Function.

    .. math::

        Log-SoftMax(x) = log(Softmax(-x))

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = LogSoftmax()(X_input)

    '''

    def __init__(self, **kwargs):
        super(LogSoftmax, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.log(K.softmax(inputs))

    def get_config(self):
        base_config = super(LogSoftmax, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


class SoftExponential(Layer):
    '''
    Soft-Exponential Activation Function.

    .. math::

        SoftExponential(x, \\alpha) = \\left\\{\\begin{matrix} - \\frac{log(1 - \\alpha(x + \\alpha))}{\\alpha}, \\alpha < 0\\\\  x, \\alpha = 0\\\\  \\frac{e^{\\alpha * x} - 1}{\\alpha} + \\alpha, \\alpha > 0 \\end{matrix}\\right.

    with 1 trainable parameter.

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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SoftExponential()(X_input)

    '''

    def __init__(self, **kwargs):
        super(SoftExponential, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight for alpha parameter. Alpha by default is initialized with random normal distribution.
        self.alpha = self.add_weight(name='alpha',
                                     initializer='random_normal',
                                     trainable=True,
                                     shape = (1,))
        super(SoftExponential, self).build(input_shape)

    def call(self, inputs):
        output =  K.cast(K.greater(self.alpha, 0), 'float32') * (K.exp(self.alpha * inputs) - 1)/(self.alpha) + \
        self.alpha + K.cast(K.less(self.alpha, 0), 'float32') * (- (K.log(1 - self.alpha * (inputs + self.alpha))) / self.alpha) + \
        K.cast(K.equal(self.alpha, 0), 'float32') * inputs

        return output

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(SoftExponential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class SReLU(Layer):
    '''
    SReLU (S-shaped Rectified Linear Activation Unit): a combination of three linear functions, which perform mapping R → R with the following formulation:

    .. math::

        h(x_i) = \\left\\{\\begin{matrix} t_i^r + a_i^r(x_i - t_i^r), x_i \\geq t_i^r \\\\  x_i, t_i^r > x_i > t_i^l\\\\  t_i^l + a_i^l(x_i - t_i^l), x_i \\leq  t_i^l \\\\ \\end{matrix}\\right.

    with 4 trainable parameters.

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Parameters:

        .. math:: \\{t_i^r, a_i^r, t_i^l, a_i^l\\}

    4 trainable parameters, which model an individual SReLU activation unit. The subscript i indicates that we allow SReLU to vary in different channels. Parameters can be initialized manually or randomly.

    References:
        - See SReLU paper:
        https://arxiv.org/pdf/1512.07030.pdf

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = SReLU(params=None)(X_input)

    '''

    def __init__(self, **kwargs):
        super(SReLU, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        '''
        Adding Trainable Parameters.
            - parameters: (tr, tl, ar, al) initialized randomly.
        '''
        self.tr = self.add_weight(name='tr',
                                 initializer='random_uniform',
                                 trainable=True,
                                 shape = (input_shape[-1],))
        self.tl = self.add_weight(name='tl',
                                 initializer='random_uniform',
                                 trainable=True,
                                 shape = (input_shape[-1],))
        self.ar = self.add_weight(name='ar',
                                 initializer='random_uniform',
                                 trainable=True,
                                 shape = (input_shape[-1],))
        self.al = self.add_weight(name='al',
                                 initializer='random_uniform',
                                 trainable=True,
                                 shape = (input_shape[-1],))

        super(SReLU, self).build(input_shape)

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, self.tr), 'float32') * (self.tr + self.ar * (inputs + self.tr)) + K.cast(K.less(inputs, self.tr), 'float32') \
               * K.cast(K.greater(inputs, self.tl), 'float32') * inputs + K.cast(K.less_equal(inputs, self.tl), 'float32') * (self.tl + self.al * (inputs + self.tl))

    def get_config(self):
        config = {'tr': float(self.tr), 't1': float(self.t1), 'ar': float(self.ar), 'a1': float(self.a1)}
        base_config = super(SReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class LeCunTanh(Layer):
    '''
    LeCun's Tanh Activation Function.

    .. math::

        LeCun's Tanh(x) = 1.7159 * tanh (\\frac{2*x}{3})

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = LeCunTanh()(X_input)

    '''

    def __init__(self, **kwargs):
        super(LeCunTanh, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return 1.7159 * K.tanh((2 * inputs)/3)

    def get_config(self):
        base_config = super(LeCunTanh, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape

class TaLU(Layer):
    '''
    TaLU Activation Function.

    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

        - Output: Same shape as the input.

    References:
        - https://github.com/mjain72/TaLuActivationFunction

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = TaLU()(X_input)

    '''

    def __init__(self, **kwargs):
        super(TaLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        cond = K.less_equal(inputs, inputs*0.0)
        t = K.tanh(inputs)
        tanH = K.tanh(-0.05)
        cond1 = K.less_equal(inputs, -0.05*(1 - inputs*0.0))
        if cond1 == True:
            y = tanH*(1 - inputs*0.0)
        else:
            y = t
        if cond == True:
            return y
        else:
            return inputs

    def get_config(self):
        base_config = super(TaLU, self).get_config()
        return dict(list(base_config.items())

    def compute_output_shape(self, input_shape):
        return input_shape


                    
class MaxoutConv2D(Layer):
    """
    References:
        - Convolution Layer followed by Maxout activation: 
        https://arxiv.org/abs/1505.03540.
        - Code :
        https://github.com/keras-team/keras/issues/8717#issue-280038650
        
    
    Parameters
    ----------
    
    kernel_size: kernel_size parameter for Conv2D
    output_dim: final number of filters after Maxout
    nb_features: number of filter maps to take the Maxout over; default=4
    padding: 'same' or 'valid'
    first_layer: True if x is the input_tensor
    input_shape: Required if first_layer=True
    
    """
    
    def __init__(self, kernel_size, output_dim, nb_features=4, padding='valid', **kwargs):
        
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.nb_features = nb_features
        self.padding = padding
        super(MaxoutConv2D, self).__init__(**kwargs)

    def call(self, x):

        output = None
        for _ in range(self.output_dim):
            
            conv_out = Conv2D(self.nb_features, self.kernel_size, padding=self.padding)(x)
            maxout_out = K.max(conv_out, axis=-1, keepdims=True)

            if output is not None:
                output = K.concatenate([output, maxout_out], axis=-1)

            else:
                output = maxout_out

        return output

    def compute_output_shape(self, input_shape):
        input_height= input_shape[1]
        input_width = input_shape[2]
        
        if(self.padding == 'same'):
            output_height = input_height
            output_width = input_width
        
        else:
            output_height = input_height - self.kernel_size[0] + 1
            output_width = input_width - self.kernel_size[1] + 1
        
        return (input_shape[0], output_height, output_width, self.output_dim)
