import tensorflow as tf
import tensorflow.keras.layers as Layer


class WeightedTanh(Layer):

    def __init__(self, input_weight):
        super(WeightedTanh, self).__init__()
        self.input_weight = input_weight
    
    def call(self, inputs):
        return tf.math.tanh(self.input_weight * inputs)
