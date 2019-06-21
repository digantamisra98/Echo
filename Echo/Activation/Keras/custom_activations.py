# -*- coding: utf-8 -*-
"""Layers that act as activation functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine.base_layer import Layer
from keras import backend as K


class mila(Layer):
    """Mila Activation Function.
    It follows:
    `f(x) =  x*tanh(ln(1+exp(\\beta + x))) = x*tanh(softplus(\\beta + x))`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        beta: scale to control the concavity of the global minima of the function (default = -0.25)
    # References
        - [Mila: Controlling Minima Concavity in Activation Function](
           https://github.com/digantamisra98/Mila)
    """

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
    """Swish Activation Function.
    It follows:
    `f(x) =  x*sigmoid(\\beta*x)`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        beta: a constant or a trainable parameter (default=1; which is equivalent to  Sigmoid-weighted Linear Unit (SiL))
    # References
        - [Searching for Activation Functions](
           https://arxiv.org/abs/1710.05941)
    """

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
    """E-Swish Activation Function.
    It follows:
    `f(x) =  \\beta*x*sigmoid(x)`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        beta: a constant parameter (default value = 1.375)
    # References
        - [E-swish: Adjusting Activations to Different Network Depths](
           https://arxiv.org/abs/1801.07145)
    """

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
    """Beta Mish Activation Function.
    It follows:
    `f(x) = x * tanh(ln((1 + exp(x))^\\beta))`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        beta: A constant or a trainable parameter (default = 1.5)
    # References
        - [β-Mish: An uni-parametric adaptive activation function derived from Mish.](
           https://github.com/digantamisra98/Beta-Mish)
    """

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
    """ISRU (Inverse Square Root Unit) Activation Function.
    It follows:
    `f(x) = x / ((1 + \\alpha * x^2)^0.5) `.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: A constant (default = 1.0)
    # References
        - [Improving Deep Learning by Inverse Square Root Linear Units (ISRLUs)](
           https://arxiv.org/abs/1710.09967)
    """

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
