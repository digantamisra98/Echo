'''
Script for demonstration of the custom Keras activation functions
implemented in the Echo package for classification of Fashion MNIST dataset.
'''
#import numpy
import numpy as np

# import utilities
import sys
sys.path.insert(0, '../')

# argsparse
import argparse

#import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from keras.datasets import fashion_mnist
from keras.utils import np_utils

# import activation functions from Echo
from echoAI.Activation.TF_Keras.custom_activation import Mila
from echoAI.Activation.TF_Keras.custom_activation import Swish
from echoAI.Activation.TF_Keras.custom_activation import ESwish
from echoAI.Activation.TF_Keras.custom_activation import ISRU
from echoAI.Activation.TF_Keras.custom_activation import BetaMish
from echoAI.Activation.TF_Keras.custom_activation import Mish
from echoAI.Activation.TF_Keras.custom_activation import SQNL
from echoAI.Activation.TF_Keras.custom_activation import FlattenTSwish
from echoAI.Activation.TF_Keras.custom_activation import ELiSH
from echoAI.Activation.TF_Keras.custom_activation import HardELiSH
from echoAI.Activation.TF_Keras.custom_activation import BentIdentity
from echoAI.Activation.TF_Keras.custom_activation import WeightedTanh
from echoAI.Activation.TF_Keras.custom_activation import SineReLU
from echoAI.Activation.TF_Keras.custom_activation import ISRLU
from echoAI.Activation.TF_Keras.custom_activation import SoftClipping
from echoAI.Activation.TF_Keras.custom_activation import Aria2
from echoAI.Activation.TF_Keras.custom_activation import CELU
from echoAI.Activation.TF_Keras.custom_activation import HardTanh
from echoAI.Activation.TF_Keras.custom_activation import LogSigmoid
from echoAI.Activation.TF_Keras.custom_activation import TanhShrink
from echoAI.Activation.TF_Keras.custom_activation import HardShrink
from echoAI.Activation.TF_Keras.custom_activation import SoftShrink
from echoAI.Activation.TF_Keras.custom_activation import SoftMin
from echoAI.Activation.TF_Keras.custom_activation import LogSoftMax
from echoAI.Activation.TF_Keras.custom_activation import SoftExponential
from echoAI.Activation.TF_Keras.custom_activation import SReLU
from echoAI.Activation.TF_Keras.custom_activation import RReLU

# activation names constants
RELU = 'relu'
MILA = 'mila'
SWISH = 'swish'
ESWISH = 'eswish'
S_ISRU = 'isru'
BETA_MISH = 'beta_mish'
MISH = 'mish'
S_SQNL = 'sqnl'
S_FTS = 'fts'
ELISH = 'elish'
HELISH = 'hard_elish'
BENTID  = 'bent_id'
WTANH = 'weighted_tanh'
SINERELU = 'sine_relu'
S_ISRLU = 'isrlu'
SC = 'soft_clipping'
ARIA2 = 'aria2'
CELU = 'celu'
HTANH = 'hard_tanh'
LSIG = 'log_sigmoid'
TANHSH = 'tanh_shrink'
HSHRINK = 'hard_shrink'
SSHRINK = 'soft_shrink'
SOFTMIN = 'softmin'
LSOFTMAX = 'log_softmax'
SEXP = 'soft_exponential'
SRELU = 'srelu'
RRELU = 'rrelu'

def main():
    '''
    Demonstrate custom activation functions to classify Fashion MNIST
    '''

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    # Add argument to choose one of the activation functions
    parser.add_argument('--activation', action='store', default = RELU,
                        help='Activation function for demonstration.',
                        choices = [SWISH, ESWISH, MILA, RELU, S_ISRU, BETA_MISH, MISH, S_SQNL, S_FTS, ELISH, HELISH, BENTID,
                        WTANH, SINERELU, S_ISRLU, SC, ARIA2, CELU, HTANH, LSIG, TANHSH, HSHRINK, SSHRINK, SOFTMIN,
                        LSOFTMAX, SEXP, SRELU,RRELU])

    # Parse command line arguments
    results = parser.parse_args()
    activation = results.activation

    # download Fasion MNIST dataset
    print('Downloading Fashion MNIST dataset: \n')
    ((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()

    # Reshape inputs
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # Normalize image vectors
    X_train = X_train.astype("float32")/255.
    X_test = X_test.astype("float32")/255.

    # one-hot encode the training and testing labels
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # initialize the label names
    labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    # initialize activation
    if (activation == SWISH):
        f_activation = Swish(beta=0.5)

    if (activation == ESWISH):
        f_activation = ESwish(beta=0.5)

    if (activation == MILA):
        f_activation = Mila(beta=0.5)

    if (activation == S_ISRU):
        f_activation = ISRU(alpha=0.5)

    if (activation == BETA_MISH):
        f_activation = BetaMish(beta=1.5)

    if (activation == MISH):
        f_activation = Mish()

    if (activation == S_SQNL):
        f_activation = SQNL()

    if (activation == S_FTS):
        f_activation = FlattenTSwish()

    if (activation == ELISH):
        f_activation = ELiSH()

    if (activation == HELISH):
        f_activation = HardELiSH()

    if (activation == BENTID):
        f_activation = BentIdentity()

    if (activation == WTANH):
        f_activation = WeightedTanh()

    if (activation == SINERELU):
        f_activation = SineReLU()

    if (activation == S_ISRLU):
        f_activation = ISRLU()

    if (activation == SC):
        f_activation = SoftClipping()

    if (activation == ARIA2):
        f_activation = Aria2()

    if (activation == CELU):
        f_activation = CELU()

    if (activation == HTANH):
        f_activation = HardTanh()

    if (activation == LSIG):
        f_activation = LogSigmoid()

    if (activation == TANHSH):
        f_activation = TanhShrink()

    if (activation == HSHRINK):
        f_activation = HardShrink()

    if (activation == SSHRINK):
        f_activation = SoftShrink()

    if (activation == SOFTMIN):
        f_activation = SoftMin()

    if (activation == LSOFTMAX):
        f_activation = LogSoftMax()

    if (activation == SEXP):
        f_activation = SoftExponential()

    if (activation == SRELU):
        f_activation = SReLU()

    if (activation == RRELU):
        f_activation = RReLU()   

    # Create model
    if activation == RELU:
        model = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')])
    else:
        model = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(128, input_shape=(784,)),
        f_activation, # use the activation function
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')])

    # Compile model
    model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = ["accuracy"])

    # Fit model
    print('Training model with {} activation function: \n'.format(activation))
    model.fit(x = X_train, y = y_train, epochs = 3, batch_size = 128)

if __name__ == '__main__':
    main()
