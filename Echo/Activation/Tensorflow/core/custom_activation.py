import tensorflow as tf
from tensorflow import math as math
from tensorflow.keras import backend as K


def weighted_tanh(inputs, weight):
    return math.tanh(inputs * weight)


def swish(inputs, beta):
    return inputs * math.sigmoid(beta * inputs)


def eswish(inputs, beta):
    return beta * inputs * math.sigmoid(inputs)


def aria2(inputs, alpha, beta):
    return math.pow(1 + math.exp(-beta * inputs), -alpha)


def elish(inputs):
    case_1 = tf.cast(math.greater_equal(inputs, 0), 'float32') * inputs * math.sigmoid(inputs)
    case_2 = tf.cast(math.less(inputs, 0), 'float32') * (math.exp(inputs) - 1) * math.sigmoid(inputs)
    return case_1 + case_2


def hard_elish(inputs):
    common = math.maximum(tf.cast(0, 'float32'), math.minimum(tf.cast(1, 'float32'), (inputs + 1) / 2))
    case_1 = tf.cast(math.greater_equal(inputs, 0), 'float32') * inputs * common
    case_2 = tf.cast(math.less(inputs, 0), 'float32') * (math.exp(inputs) - 1) * common
    return case_1 + case_2


def mila(inputs):
    return inputs * math.tanh(math.log(1 + math.exp(beta + inputs)))


def sine_relu(inputs, epsilon):
    case_1 = tf.cast(math.greater(inputs, 0), 'float32') * inputs
    case_2 = tf.cast(math.less_equal(epsilon * (math.sin(inputs) - math.cos(inputs)), 0), 'float32')
    return case_1 + case_2


def flatten_tswish(inputs):
    return tf.cast(math.greater_equal(inputs, 0), 'float32') * inputs * math.sigmoid(inputs)


def sqnl(inputs):
    case_1 = tf.cast(math.greater(inputs, 2), 'float32')
    case_2 = tf.cast(math.greater_equal(inputs, 0), 'float32') * tf.cast(math.less_equal(inputs, 2), 'float32') * (inputs - math.pow(inputs, 2) / tf.cast(4, 'float32'))
    case_3 = tf.cast(math.greater_equal(inputs, -2), 'float32') * tf.cast(math.less(inputs, 0), 'float32') * (inputs + math.pow(inputs, 2) / tf.cast(4, 'float32'))
    case_4 = tf.cast(math.less(inputs, -2), 'float32') * tf.cast(-1, 'float32')
    return case_1 + case_2 + case_3 + case_4


def isru(inputs):
    return inputs / math.sqrt(1 + alpha * math.pow(inputs, 2))


def isrlu(inputs, alpha):
    case_1 = tf.cast(math.greater_equal(inputs, 0), 'float32') * inputs
    case_2 = tf.cast(math.less(inputs, 0), 'float32') * inputs  / math.sqrt(1 + alpha * math.pow(inputs, 2))
    return case_1 + case_2


def bents_identity(inputs):
    return inputs + (math.sqrt(math.pow(inputs, 2) + 1) - 1) / 2


def soft_clipping(inputs):
    return tf.math.log((1 + math.exp(alpha * inputs)) * math.sigmoid(alpha * (1 - inputs)))


def srelu(inputs, t, a, r, l):
    condition_1 = tf.cast(math.greater_equal(inputs, math.pow(t, r)), 'float32')
    condition_2 = tf.cast(math.greater(tf.math.pow(t, r), inputs), 'float32') + tf.cast(math.greater(inputs, math.pow(t, l)), 'float32')
    condition_3 = tf.cast(math.less_equal(inputs, math.pow(t, l)), 'float32')
    case_1 = condition_1 * (math.pow(t, r) + math.pow(a, r) * (inputs - math.pow(t, r)))
    case_2 = condition_2 * inputs
    case_3 = condition_3 * (math.pow(t, l) + math.pow(a, l) * (inputs - math.pow(t, l)))
    return case_1 + case_2 + case_3


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


def soft_exponential(inputs):
    condition_1 = tf.cast(math.less(alpha, 0), 'float32')
    condition_2 = tf.cast(math.equal(alpha, 0), 'float32')
    condition_3 = tf.cast(math.greater(alpha, 0), 'float32')
    case_1 = condition_1 * (-1 / alpha) * math.log(1 - alpha * (inputs + alpha))
    case_2 = condition_2 * inputs
    case_3 = condition_3 * (alpha + (1 / alpha) * (math.exp(alpha * inputs) - 1))
    return case_1 + case_2 + case_3


def maxout(inputs):
    return K.max(inputs)


def mish(inputs):
    return inputs * math.tanh(math.log(1 + math.exp(inputs)))


def beta_mish(inputs, beta):
    return inputs * math.tanh(math.log(math.pow(1 + math.exp(inputs), beta)))


def celu(inputs, alpha):
    case_1 = tf.cast(math.greater_equal(inputs, 0), 'float32') * inputs
    case_2 = tf.cast(math.less(inputs, 0), 'float32') * alpha * (math.exp(inputs / alpha) - 1)
    return case_1 + case_2


def relu6(inputs):
    return math.minimum(math.maximum(tf.cast(0, 'float32'), inputs), 6)


def hard_tanh(inputs):
    case_1 = tf.cast(math.greater(inputs, 1), 'float32')
    case_2 = tf.cast(math.less(inputs, -1), 'float32') * -1
    case_3 = tf.cast(math.greater_equal(inputs, -1), 'float32') * tf.cast(math.less_equal(inputs, 1), 'float32') * inputs
    return case_1 + case_2 + case_3


def log_sigmoid(inputs):
    return math.log(tf.math.sigmoid(inputs))


def tanh_shrink(inputs):
    return inputs - math.tanh(inputs)


def hard_shrink(inputs, _lambda = 0.6):
    case_1 = tf.cast(math.greater(inputs, _lambda), 'float32') * inputs
    case_2 = tf.cast(math.less(inputs, -1 * _lambda), 'float32') * inputs
    return case_1 + case_2


def  softshrink(inputs, _lambda = 0.5):
    case_1 = tf.cast(math.greater(inputs, _lambda), 'float32') * (inputs - _lambda)
    case_2 = tf.cast(math.less(inputs, -1 * _lambda), 'float32') * (inputs - _lambda)
    return case_1 + case_2


def softmin(inputs):
    return math.softmax(-inputs)


def log_softmin(inputs):
    return math.log(math.softmax(inputs))