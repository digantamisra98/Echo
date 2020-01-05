"""
Script for demonstration of the custom Keras activation functions
implemented in the Echo package for classification of Fashion MNIST dataset.
"""

# import utilities
import sys

# argsparse
import argparse

# import keras
from keras.layers import (
    Input,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
)
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.datasets import fashion_mnist
from keras.utils import np_utils

# import activation functions from Echo
from echoAI.Activation.Keras.custom_activations import Mila
from echoAI.Activation.Keras.custom_activations import Swish
from echoAI.Activation.Keras.custom_activations import Eswish
from echoAI.Activation.Keras.custom_activations import ISRU
from echoAI.Activation.Keras.custom_activations import BetaMish
from echoAI.Activation.Keras.custom_activations import Mish
from echoAI.Activation.Keras.custom_activations import SQNL
from echoAI.Activation.Keras.custom_activations import FTS
from echoAI.Activation.Keras.custom_activations import Elish
from echoAI.Activation.Keras.custom_activations import HardElish
from echoAI.Activation.Keras.custom_activations import BentID
from echoAI.Activation.Keras.custom_activations import WeightedTanh
from echoAI.Activation.Keras.custom_activations import SineReLU
from echoAI.Activation.Keras.custom_activations import ISRLU
from echoAI.Activation.Keras.custom_activations import SoftClipping
from echoAI.Activation.Keras.custom_activations import Aria2
from echoAI.Activation.Keras.custom_activations import Celu
from echoAI.Activation.Keras.custom_activations import ReLU6
from echoAI.Activation.Keras.custom_activations import HardTanh
from echoAI.Activation.Keras.custom_activations import LogSigmoid
from echoAI.Activation.Keras.custom_activations import TanhShrink
from echoAI.Activation.Keras.custom_activations import HardShrink
from echoAI.Activation.Keras.custom_activations import SoftShrink
from echoAI.Activation.Keras.custom_activations import SoftMin
from echoAI.Activation.Keras.custom_activations import LogSoftmax
from echoAI.Activation.Keras.custom_activations import SoftExponential
from echoAI.Activation.Keras.custom_activations import SReLU

sys.path.insert(0, "../")

# activation names constants
RELU = "relu"
MILA = "mila"
SWISH = "swish"
ESWISH = "eswish"
S_ISRU = "isru"
BETA_MISH = "beta_mish"
MISH = "mish"
S_SQNL = "sqnl"
S_FTS = "fts"
ELISH = "elish"
HELISH = "hard_elish"
BENTID = "bent_id"
WTANH = "weighted_tanh"
SINERELU = "sine_relu"
S_ISRLU = "isrlu"
SC = "soft_clipping"
ARIA2 = "aria2"
CELU = "celu"
RELU6 = "relu6"
HTANH = "hard_tanh"
LSIG = "log_sigmoid"
TANHSH = "tanh_shrink"
HSHRINK = "hard_shrink"
SSHRINK = "soft_shrink"
SOFTMIN = "softmin"
LSOFTMAX = "log_softmax"
SEXP = "soft_exponential"
SRELU = "srelu"


def CNNModel(input_shape, activation="relu"):
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
    X = Conv2D(32, (3, 3), strides=(1, 1), name="conv0")(X)
    X = BatchNormalization(axis=3, name="bn0")(X)

    if activation == RELU:
        X = Activation("relu")(X)

    if activation == SWISH:
        X = Swish(beta=0.5)(X)

    if activation == ESWISH:
        X = Eswish(beta=0.5)(X)

    if activation == MILA:
        X = Mila(beta=0.5)(X)

    if activation == S_ISRU:
        X = ISRU(alpha=0.5)(X)

    if activation == BETA_MISH:
        X = BetaMish(beta=1.5)(X)

    if activation == MISH:
        X = Mish()(X)

    if activation == S_SQNL:
        X = SQNL()(X)

    if activation == S_FTS:
        X = FTS()(X)

    if activation == ELISH:
        X = Elish()(X)

    if activation == HELISH:
        X = HardElish()(X)

    if activation == BENTID:
        X = BentID()(X)

    if activation == WTANH:
        X = WeightedTanh()(X)

    if activation == SINERELU:
        X = SineReLU()(X)

    if activation == S_ISRLU:
        X = ISRLU()(X)

    if activation == SC:
        X = SoftClipping()(X)

    if activation == ARIA2:
        X = Aria2()(X)

    if activation == CELU:
        X = Celu()(X)

    if activation == RELU6:
        X = ReLU6()(X)

    if activation == HTANH:
        X = HardTanh()(X)

    if activation == LSIG:
        X = LogSigmoid()(X)

    if activation == TANHSH:
        X = TanhShrink()(X)

    if activation == HSHRINK:
        X = HardShrink()(X)

    if activation == SSHRINK:
        X = SoftShrink()(X)

    if activation == SOFTMIN:
        X = SoftMin()(X)

    if activation == LSOFTMAX:
        X = LogSoftmax()(X)

    if activation == SEXP:
        X = SoftExponential()(X)

    if activation == SRELU:
        X = SReLU()(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name="max_pool")(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation="softmax", name="fc")(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name="CNNModel")

    return model


def main():
    """
    Demonstrate custom activation functions to classify Fashion MNIST
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Argument parser")

    # Add argument to choose one of the activation functions
    parser.add_argument(
        "--activation",
        action="store",
        default=RELU,
        help="Activation function for demonstration.",
        choices=[
            SWISH,
            ESWISH,
            MILA,
            RELU,
            S_ISRU,
            BETA_MISH,
            MISH,
            S_SQNL,
            S_FTS,
            ELISH,
            HELISH,
            BENTID,
            WTANH,
            SINERELU,
            S_ISRLU,
            SC,
            ARIA2,
            CELU,
            RELU6,
            HTANH,
            LSIG,
            TANHSH,
            HSHRINK,
            SSHRINK,
            SOFTMIN,
            LSOFTMAX,
            SEXP,
            SRELU,
        ],
    )

    # Parse command line arguments
    results = parser.parse_args()
    activation = results.activation

    # download Fasion MNIST dataset
    print("Downloading Fashion MNIST dataset: \n")
    ((X_train, y_train), (X_test, y_test)) = fashion_mnist.load_data()

    # Reshape inputs
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    # Normalize image vectors
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # one-hot encode the training and testing labels
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    # initialize the label names
    labelNames = [
        "top",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]

    # Create model
    model = CNNModel((28, 28, 1), activation)

    # Compile model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    # Fit model
    print("Training model with {} activation function: \n".format(activation))
    model.fit(x=X_train, y=y_train, epochs=3, batch_size=128)


if __name__ == "__main__":
    main()
