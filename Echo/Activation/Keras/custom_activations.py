# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K
from keras import activations
from keras.layers import Wrapper, Lambda
from keras import initializers

class mila(Layer):
    '''
    Mila Activation Function.

    .. math::

        mila(x) = x * tanh(ln(1 + e^{\\beta + x})) = x * tanh(softplus(\\beta + x)

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
        >>> X = mila(beta=0.5)(X_input)

    '''

    def __init__(self, beta=-0.25, **kwargs):
        super(mila, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs*K.tanh(K.softplus(inputs + self.beta))

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(mila, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class swish(Layer):
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
        >>> X = swish(beta=0.5)(X_input)

    '''

    def __init__(self, beta=1.0, **kwargs):
        super(swish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs*K.sigmoid(inputs*self.beta)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(swish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class eswish(Layer):
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
        >>> X = eswish(beta=0.5)(X_input)

    '''

    def __init__(self, beta=1.375, **kwargs):
        super(eswish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return self.beta*inputs*K.sigmoid(inputs)

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(eswish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class beta_mish(Layer):
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
        >>> X = beta_mish(beta=1.5)(X_input)

    '''

    def __init__(self, beta=1.5, **kwargs):
        super(beta_mish, self).__init__(**kwargs)
        self.supports_masking = True
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return inputs*K.tanh(K.log(K.pow((1+K.exp(inputs)),self.beta)))

    def get_config(self):
        config = {'beta': float(self.beta)}
        base_config = super(beta_mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class isru(Layer):
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
        >>> X = isru(alpha=0.5)(X_input)

    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(isru, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return inputs/(K.sqrt(1 + self.alpha * K.pow(inputs,2)))

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(isru, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class mish(Layer):
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

    Examples:
        >>> X_input = Input(input_shape)
        >>> X = mish()(X_input)

    '''

    def __init__(self, **kwargs):
        super(mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        base_config = super(mish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class sqnl(Layer):
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
        >>> X = sqnl()(X_input)

    '''

    def __init__(self, **kwargs):
        super(sqnl, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater(inputs , 2), 'float32')\
        + (inputs - K.pow(inputs,2)/4) * K.cast(K.greater_equal(inputs,0), 'float32') * K.cast(K.less_equal(inputs, 2), 'float32') \
        + (inputs + K.pow(inputs,2)/4) * K.cast(K.less(inputs, 0), 'float32') * K.cast(K.greater_equal(inputs, -2), 'float32') - K.cast(K.less(inputs, -2), 'float32')

    def get_config(self):
        base_config = super(sqnl, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class fts(Layer):
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
        >>> X = fts()(X_input)

    '''

    def __init__(self, **kwargs):
        super(fts, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs / (1 + K.exp(- inputs))

    def get_config(self):
        base_config = super(fts, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class elish(Layer):
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
        >>> X = elish()(X_input)

    '''

    def __init__(self, **kwargs):
        super(elish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs * K.sigmoid(inputs) + K.cast(K.less(inputs, 0), 'float32') * (K.exp(inputs) - 1) / (K.exp(- inputs) + 1)

    def get_config(self):
        base_config = super(elish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class hard_elish(Layer):
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
        >>> X = hard_elish()(X_input)

    '''

    def __init__(self, **kwargs):
        super(hard_elish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs * K.maximum(K.cast_to_floatx(0.0), K.minimum(K.cast_to_floatx(1.0), (inputs + 1.0)/2.0)) \
        + K.cast(K.less(inputs, 0), 'float32') * (K.exp(inputs - 1) * K.maximum(K.cast_to_floatx(0.0), K.minimum(K.cast_to_floatx(1.0), (inputs + 1.0)/2.0)))

    def get_config(self):
        base_config = super(hard_elish, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class bent_id(Layer):
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
        >>> X = bent_id()(X_input)

    '''

    def __init__(self, **kwargs):
        super(bent_id, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs + ((K.sqrt(K.pow(inputs,2)+1)-1)/2)

    def get_config(self):
        base_config = super(bent_id, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class weighted_tanh(Layer):
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
        >>> X = weighted_tanh(weight=1.0)(X_input)

    '''

    def __init__(self, weight=1.0, **kwargs):
        super(weighted_tanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.weight = K.cast_to_floatx(weight)

    def call(self, inputs):
        return K.tanh(inputs * self.weight)

    def get_config(self):
        config = {'weight': float(self.weight)}
        base_config = super(weighted_tanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class sineReLU(Layer):
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
        >>> X = sineReLU(epsilon=0.01)(X_input)

    '''

    def __init__(self, epsilon=0.01, **kwargs):
        super(sineReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = K.cast_to_floatx(epsilon)

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs + K.cast(K.less(inputs, 0), 'float32') * self.epsilon * (K.sin(inputs) - K.cos(inputs))

    def get_config(self):
        config = {'epsilon': float(self.epsilon)}
        base_config = super(sineReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class isrlu(Layer):
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
        >>> X = isrlu(alpha=1.0)(X_input)

    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(isrlu, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.cast(K.less(inputs, 0), 'float32') * isru(alpha=self.alpha)(inputs) + K.cast(K.greater_equal(inputs, 0), 'float32') * inputs

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(isrlu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class soft_clipping(Layer):
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
        >>> X =soft_clipping(alpha=0.5)(X_input)

    '''

    def __init__(self, alpha=0.5, **kwargs):
        super(soft_clipping, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return (1 / self.alpha) * K.log((1 + K.exp(self.alpha * inputs))/(1 + K.exp(self.alpha *(inputs - 1))))

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(soft_clipping, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class aria2(Layer):
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
        >>> X =aria2(alpha=1.0, beta=0.5)(X_input)

    '''

    def __init__(self, alpha=1.0, beta=0.5, **kwargs):
        super(aria2, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
        self.beta = K.cast_to_floatx(beta)

    def call(self, inputs):
        return K.pow((1 + K.exp(-self.beta * inputs)), -self.alpha)

    def get_config(self):
        config = {'alpha': float(self.alpha), 'beta': float(self.beta)}
        base_config = super(aria2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class celu(Layer):
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
        >>> X = celu(alpha=1.0)(X_input)

    '''

    def __init__(self, alpha=1.0, **kwargs):
        super(celu, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 0), 'float32') * inputs + K.cast(K.less(inputs, 0), 'float32') * self.alpha * (K.exp (inputs / self.alpha) - 1)

    def get_config(self):
        config = {'alpha': float(self.alpha)}
        base_config = super(celu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class relu6(Layer):
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
        >>> X = relu6()(X_input)

    '''

    def __init__(self, **kwargs):
        super(relu6, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, 6), 'float32') * 6 + K.cast(K.less(inputs, 6), 'float32') * K.relu(inputs)

    def get_config(self):
        base_config = super(relu6, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class hard_tanh(Layer):
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
        >>> X = hard_tanh()(X_input)

    '''

    def __init__(self, **kwargs):
        super(hard_tanh, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.cast(K.greater(inputs , 1), 'float32')\
        + inputs * K.cast(K.less_equal(inputs, 1), 'float32') * K.cast(K.greater_equal(inputs, -1), 'float32') - K.cast(K.less(inputs, -1), 'float32')

    def get_config(self):
        base_config = super(hard_tanh, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class log_sigmoid(Layer):
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
        >>> X = log_sigmoid()(X_input)

    '''

    def __init__(self, **kwargs):
        super(log_sigmoid, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.log(K.sigmoid(inputs))

    def get_config(self):
        base_config = super(log_sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class tanh_shrink(Layer):
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
        >>> X = tanh_shrink()(X_input)

    '''

    def __init__(self, **kwargs):
        super(tanh_shrink, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs - K.tanh(inputs)

    def get_config(self):
        base_config = super(tanh_shrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class hard_shrink(Layer):
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
        >>> X = hard_shrink()(X_input)

    '''

    def __init__(self, lambd = 0.5, **kwargs):
        super(hard_shrink, self).__init__(**kwargs)
        self.supports_masking = True
        self.lambd = K.cast_to_floatx(lambd)

    def call(self, inputs):
        return K.cast(K.greater(inputs , self.lambd), 'float32') * inputs \
        + 0 * K.cast(K.less_equal(inputs, self.lambd), 'float32') * K.cast(K.greater_equal(inputs, -self.lambd), 'float32') + inputs *  K.cast(K.less(inputs, -self.lambd), 'float32')

    def get_config(self):
        config = {'lambd': float(self.lambd)}
        base_config = super(hard_shrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class soft_shrink(Layer):
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
        >>> X = soft_shrink()(X_input)

    '''

    def __init__(self, lambd= 0.5, **kwargs):
        super(soft_shrink, self).__init__(**kwargs)
        self.supports_masking = True
        self.lambd = K.cast_to_floatx(lambd)

    def call(self, inputs):
        return (K.cast(K.greater(inputs , self.lambd), 'float32') * (inputs - self.lambd)) \
        + (0 * K.cast(K.less_equal(inputs, self.lambd), 'float32') * K.cast(K.greater_equal(inputs, -self.lambd), 'float32')) \
        + ((inputs + self.lambd) *  K.cast(K.less(inputs, -self.lambd), 'float32'))

    def get_config(self):
        config = {'lambd': float(self.lambd)}
        base_config = super(soft_shrink, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class softmin(Layer):
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
        >>> X = softmin()(X_input)

    '''

    def __init__(self, **kwargs):
        super(softmin, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.softmax(-inputs)

    def get_config(self):
        base_config = super(softmin, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class log_softmax(Layer):
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
        >>> X = log_softmax()(X_input)

    '''

    def __init__(self, **kwargs):
        super(log_softmax, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return K.log(K.softmax(inputs))

    def get_config(self):
        base_config = super(log_softmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class soft_exponential(Layer):
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
        >>> X = soft_exponential()(X_input)

    '''

    def __init__(self, **kwargs):
        super(soft_exponential, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        # Create a trainable weight for alpha parameter. Alpha by default is initialized with random normal distribution.
        self.alpha = self.add_weight(name='alpha',
                                     initializer='random_normal',
                                     trainable=True,
                                     shape = (1,))
        super(soft_exponential, self).build(input_shape)

    def call(self, inputs):
        output =  K.cast(K.greater(self.alpha, 0), 'float32') * (K.exp(self.alpha * inputs) - 1)/(self.alpha) + \
        self.alpha + K.cast(K.less(self.alpha, 0), 'float32') * (- (K.log(1 - self.alpha * (inputs + self.alpha))) / self.alpha) + \
        K.cast(K.equal(self.alpha, 0), 'float32') * inputs

        return output

    def get_config(self):
        base_config = super(soft_exponential, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

class srelu(Layer):
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
        >>> X = srelu(params=None)(X_input)

    '''

    def __init__(self, **kwargs):
        super(srelu, self).__init__(**kwargs)
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

        super(srelu, self).build(input_shape)

    def call(self, inputs):
        return K.cast(K.greater_equal(inputs, self.tr), 'float32') * (self.tr + self.ar * (inputs + self.tr)) + K.cast(K.less(inputs, self.tr), 'float32') \
               * K.cast(K.greater(inputs, self.tl), 'float32') * inputs + K.cast(K.less_equal(inputs, self.tl), 'float32') * (self.tl + self.al * (inputs + self.tl))

    def get_config(self):
        base_config = super(srelu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class BReLU(Layer):

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


class GatedConvBlock(Wrapper):
    
    def __init__(self, conv_layer, conv_num = 3, gate_activation='sigmoid', **kwargs):
        super(GatedConvBlock, self).__init__(conv_layer,**kwargs)
        self.conv_num= conv_num
        self.gate_activation = activations.get(gate_activation)
        self.conv_layers = []
        self.input_spec = conv_layer.input_spec 
        self.rank = conv_layer.rank
        if conv_layer.padding != 'same':
            raise ValueError("The padding mode of this layer must be `same`, But found `{}`".format(self.padding))
        self.filters = conv_layer.filters//2
        #create conv layers 
        import copy
        for i in range(self.conv_num):
            new_conv_layer = copy.deepcopy(conv_layer)
            new_conv_layer.name = 'GatedConvBlock_{}_{}'.format(conv_layer.name, i)
            self.conv_layers.append(new_conv_layer) 
    
    def build(self, input_shape):
        if self.conv_layers[0].filters != input_shape[-1]*2:
            raise ValueError("For efficient, the sub-conv-layer's filters must be the twice of input_shape[-1].\nBut found filters={},input_shape[-1]={}".format(self.conv_layers[0].filters, input_shape[-1]))
        input_shape_current = input_shape
        for layer in self.conv_layers:
            with K.name_scope(layer.name):
                layer.build(input_shape_current)
            input_shape_current = input_shape
        self.built = True            
        pass
    
    def compute_output_shape(self, input_shape):
        input_shape_current = input_shape
        for layer in self.conv_layers:
            input_shape_current = layer.compute_output_shape(input_shape_current)
            output_shape = list(input_shape_current)
            output_shape[-1] = int(output_shape[-1]/2)
            input_shape_current = output_shape   
        return tuple(input_shape_current)
    
    def half_slice(self, x):
        ndim = self.rank +2
        if ndim ==3:
            linear_output = x[:,:,:self.filters]
            gated_output = x[:,:,self.filters:]
        elif ndim ==4:
            linear_output = x[:,:,:,:self.filters]
            gated_output = x[:,:,:,self.filters:]
        elif ndim ==5:
            linear_output = x[:,:,:,:,:self.filters]
            gated_output = x[:,:,:,:,self.filters:]
        else:
            raise ValueError("This class only support for 1D, 2D, 3D conv, but the input's ndim={}".format(ndim))
        return linear_output, gated_output
    
    def call(self, inputs):
        input_current = inputs  
        for i,layer in enumerate(self.conv_layers):
            output_current = layer(inputs= input_current) 
            linear_output, gated_output = self.half_slice(output_current)
            input_current = linear_output*self.gate_activation(gated_output)
            input_current._keras_shape = K.int_shape(linear_output)
        #residual connection
        output = input_current + inputs
        return output
    
    def get_weights(self):
        weights = None 
        for layer in self.conv_layers:
            weights += layer.get_weights()
        return weights
    
    def set_weights(self, weights):
        for layer in self.conv_layers:
            layer.set_weights(weights)
        pass
    
    @property
    def trainable_weights(self):
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'trainable_weights'):
                weights += layer.trainable_weights
        return weights
        pass
    
    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.conv_layers:
            if hasattr(layer, 'non_trainable_weights'):
                weights += layer.non_trainable_weights
        return weights
        pass
    
    @property
    def updates(self):
        updates_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'updates'):
                updates_ += layer.upates
        return updates_
        pass
    
    @property
    def losses(self):
        losses_ = []
        for layer in self.conv_layers:
            if hasattr(layer, 'losses'):
                losses_ += layer.losses
        return losses_
        pass
    
    @property
    def constraints(self):
        constraints_ = {}
        for layer in self.conv_layers:
            if hasattr(layer, 'constraints'):
                constraints_.update(layer.constraints)
        return constraints_