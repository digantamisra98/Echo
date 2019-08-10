import tensorflow as tf
from tensorflow.keras.layers import Layer


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