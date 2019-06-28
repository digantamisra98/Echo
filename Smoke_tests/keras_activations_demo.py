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
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.utils import np_utils

# import activation functions from Echo
from Echo.Activation.Keras.custom_activations import mila
from Echo.Activation.Keras.custom_activations import swish
from Echo.Activation.Keras.custom_activations import eswish
from Echo.Activation.Keras.custom_activations import isru
from Echo.Activation.Keras.custom_activations import beta_mish
from Echo.Activation.Keras.custom_activations import mish
from Echo.Activation.Keras.custom_activations import sqnl
from Echo.Activation.Keras.custom_activations import fts
from Echo.Activation.Keras.custom_activations import elish
from Echo.Activation.Keras.custom_activations import hard_elish
from Echo.Activation.Keras.custom_activations import bent_id

# activation names constants
RELU = 'relu'
MILA = 'mila'
SWISH = 'swish'
ESWISH = 'eswish'
ISRU = 'isru'
BETA_MISH = 'beta_mish'
MISH = 'mish'
SQNL = 'sqnl'
FTS = 'fts'
ELISH = 'elish'
HELISH = 'hard_elish'
BENTID  = 'bent_id'

def CNNModel(input_shape, activation = 'relu'):
    """
    Implementation of the simple CNN.

    INPUT:
        input_shape -- shape of the images of the dataset

    OUTPUT::
        model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)

    if (activation == RELU):
        X = Activation('relu')(X)

    if (activation == SWISH):
        X = swish(beta=0.5)(X)

    if (activation == ESWISH):
        X = eswish(beta=0.5)(X)

    if (activation == MILA):
        X = mila(beta=0.5)(X)

    if (activation == ISRU):
        X = isru(alpha=0.5)(X)

    if (activation == BETA_MISH):
        X = beta_mish(beta=1.5)(X)

    if (activation == MISH):
        X = mish()(X)

    if (activation == SQNL):
        X = sqnl()(X)

    if (activation == FTS):
        X = fts()(X)

    if (activation == ELISH):
        X = elish()(X)

    if (activation == HELISH):
        X = hard_elish()(X)

    if (activation == BENTID):
        X = bent_id()(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation='softmax', name='fc')(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='CNNModel')

    return model

def main():
    '''
    Demonstrate custom activation functions to classify Fashion MNIST
    '''

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    # Add argument to choose one of the activation functions
    parser.add_argument('--activation', action='store', default = RELU,
                        help='Activation function for demonstration.',
                        choices = [SWISH, ESWISH, MILA, RELU, ISRU, BETA_MISH, MISH, SQNL, FTS, ELISH, HELISH, BENTID])

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

    # Create model
    model = CNNModel((28,28,1), activation)

    # Compile model
    model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = ["accuracy"])

    # Fit model
    print('Training model with {} activation function: \n'.format(activation))
    model.fit(x = X_train, y = y_train, epochs = 3, batch_size = 128)

if __name__ == '__main__':
    main()
