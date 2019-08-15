import tensorflow as tf
from tensorflow.keras.layers import Layer, Lambda, InputSpec
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class WeightedTanh(Layer):

    def __init__(self, input_weight):
        super(WeightedTanh, self).__init__()
        self.input_weight = input_weight
    
    def call(self, inputs):
        return tf.math.tanh(self.input_weight * inputs)


class Swish(Layer):

    def __init__(self, beta):
        super(Swish, self).__init__()
        self.beta = beta
    
    def call(self, inputs):
        return inputs * tf.math.sigmoid(self.beta * inputs)


class ESwish(Layer):

    def __init__(self, beta):
        super(ESwish, self).__init__()
        self.beta = beta
    
    def call(self, inputs):
        return self.beta * inputs * tf.math.sigmoid(inputs)


class Aria2(Layer):

    def __init__(self, alpha, beta):
        super(Aria2, self).__init__()
        self.alpha = alpha
        self.beta = beta
    
    def call(self, inputs):
        return tf.math.pow(1 + tf.math.exp(-self.beta * inputs), -self.alpha)


class Mila(Layer):

    def __init__(self, beta):
        super(Mila, self).__init__()
        self.beta = beta
    
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.log(1 + tf.math.exp(self.beta + inputs)))


class ISRU(Layer):

    def __init__(self, alpha):
        super(ISRU, self).__init__()
        self.alpha = alpha
    
    def call(self, inputs):
        return inputs / tf.math.sqrt(1 + self.alpha * tf.math.pow(inputs, 2))


class BentIdentity(Layer):

    def __init__(self):
        super(BentIdentity, self).__init__()
    
    def call(self, inputs):
        return inputs + (tf.math.sqrt(tf.math.pow(inputs, 2) + 1) - 1) / 2


class SoftClipping(Layer):

    def __init__(self, alpha):
        super(SoftClipping, self).__init__()
        self.alpha = alpha
    
    def call(self, inputs):
        return tf.math.log((1 + tf.math.exp(self.alpha * inputs)) * tf.math.sigmoid(self.alpha * (1 - inputs)))


class Mish(Layer):

    def __init__(self):
        super(Mish, self).__init__()
    
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.log(1 + tf.math.exp(inputs)))


class BetaMish(Layer):

    def __init__(self, beta):
        super(BetaMish, self).__init__()
        self.beta = beta
    
    def call(self, inputs):
        return inputs * tf.math.tanh(tf.math.log(tf.math.pow(1 + tf.math.exp(inputs), self.beta)))


class ELiSH(Layer):

    def __init__(self):
        super(ELiSH, self).__init__()
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs * tf.math.sigmoid(inputs)
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * (tf.math.exp(inputs) - 1) * tf.math.sigmoid(inputs)
        return case_1 + case_2


class HardELiSH(Layer):

    def __init__(self):
        super(HardELiSH, self).__init__()
    
    def call(self, inputs):
        common = tf.math.maximum(tf.cast(0, 'float32'), tf.math.minimum(tf.cast(1, 'float32'), (inputs + 1) / 2))
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs * common
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * (tf.math.exp(inputs) - 1) * common
        return case_1 + case_2


class SineReLU(Layer):

    def __init__(self, epsilon):
        super(SineReLU, self).__init__()
        self.epsilon = epsilon
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, 0), 'float32') * inputs
        case_2 = tf.cast(tf.math.less_equal(self.epsilon * (tf.math.sin(inputs) - tf.math.cos(inputs)), 0), 'float32')
        return case_1 + case_2


class FlattenTSwish(Layer):

    def __init__(self):
        super(FlattenTSwish, self).__init__()
    
    def call(self, inputs):
        return tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs * tf.math.sigmoid(inputs)


class SQNL(Layer):

    def __init__(self):
        super(SQNL, self).__init__()
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, 2), 'float32')
        case_2 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * tf.cast(tf.math.less_equal(inputs, 2), 'float32') * (inputs - tf.math.pow(inputs, 2) / tf.cast(4, 'float32'))
        case_3 = tf.cast(tf.math.greater_equal(inputs, -2), 'float32') * tf.cast(tf.math.less(inputs, 0), 'float32') * (inputs + tf.math.pow(inputs, 2) / tf.cast(4, 'float32'))
        case_4 = tf.cast(tf.math.less(inputs, -2), 'float32') * tf.cast(-1, 'float32')
        return case_1 + case_2 + case_3 + case_4


class ISRLU(Layer):

    def __init__(self, alpha):
        super(ISRLU, self).__init__()
        self.alpha = alpha
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * inputs  / tf.math.sqrt(1 + self.alpha * tf.math.pow(inputs, 2))
        return case_1 + case_2


class SoftExponential(Layer):

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

    def __init__(self, alpha):
        super(CELU, self).__init__()
        self.alpha = alpha
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater_equal(inputs, 0), 'float32') * inputs
        case_2 = tf.cast(tf.math.less(inputs, 0), 'float32') * self.alpha * (tf.math.exp(inputs / self.alpha) - 1)
        return case_1 + case_2


class HardTanh(Layer):

    def __init__(self):
        super(HardTanh, self).__init__()
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, 1), 'float32')
        case_2 = tf.cast(tf.math.less(inputs, -1), 'float32') * -1
        case_3 = tf.cast(tf.math.greater_equal(inputs, -1), 'float32') * tf.cast(tf.math.less_equal(inputs, 1), 'float32') * inputs
        return case_1 + case_2 + case_3


class LogSigmoid(Layer):

    def __init__(self):
        super(LogSigmoid, self).__init__()
    
    def call(self, inputs):
        return tf.math.log(tf.math.sigmoid(inputs))


class TanhShrink(Layer):

    def __init__(self):
        super(TanhShrink, self).__init__()
    
    def call(self, inputs):
        return inputs - tf.math.tanh(inputs)


class HardShrink(Layer):

    def __init__(self, _lambda = 0.5):
        super(HardShrink, self).__init__()
        self._lambda = _lambda
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, self._lambda), 'float32') * inputs
        case_2 = tf.cast(tf.math.less(inputs, -1 * self._lambda), 'float32') * inputs
        return case_1 + case_2


class SoftShrink(Layer):

    def __init__(self, _lambda = 0.5):
        super(HardShrink, self).__init__()
        self._lambda = _lambda
    
    def call(self, inputs):
        case_1 = tf.cast(tf.math.greater(inputs, self._lambda), 'float32') * (inputs - self._lambda)
        case_2 = tf.cast(tf.math.less(inputs, -1 * self._lambda), 'float32') * (inputs - self._lambda)
        return case_1 + case_2


class SoftMin(Layer):

    def __init__(self):
        super(SoftMin, self).__init__()
    
    def call(self, inputs):
        return tf.math.softmax(-inputs)


class LogSoftMax(Layer):

    def __init__(self):
        super(LogSoftMax, self).__init__()
    
    def call(self, inputs):
        return tf.math.log(tf.math.softmax(inputs))


class MaxOut(Layer):

    def __init__(self):
        super(MaxOut, self).__init__()
    
    def call(self, inputs):
        return K.max(inputs)


class SReLU(Layer):

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


class ReLU6(Layer):

    def __init__(self):
        super(ReLU6, self).__init__()
    
    def call(inputs):
        return tf.math.minimum(tf.math.maximum(tf.cast(0, 'float32'), inputs), 6)