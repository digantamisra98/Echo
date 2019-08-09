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