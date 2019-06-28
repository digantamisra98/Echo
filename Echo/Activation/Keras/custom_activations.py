# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K


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
