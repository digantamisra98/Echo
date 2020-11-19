"""
Script for demonstration of the custom activation functions
implemented in the EchoAI package for classification of Fashion MNIST dataset.
"""

# import custom packages
import argparse
import sys

# import basic libraries
from collections import OrderedDict

# import pytorch
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms

import echoAI.Activation.Torch.functional as Func
from echoAI.Activation.Torch.aria2 import Aria2
from echoAI.Activation.Torch.bent_id import BentID
from echoAI.Activation.Torch.beta_mish import BetaMish
from echoAI.Activation.Torch.elish import Elish
from echoAI.Activation.Torch.eswish import Eswish
from echoAI.Activation.Torch.fts import FTS
from echoAI.Activation.Torch.hard_elish import HardElish
from echoAI.Activation.Torch.isrlu import ISRLU
from echoAI.Activation.Torch.isru import ISRU
from echoAI.Activation.Torch.mila import Mila
from echoAI.Activation.Torch.mish import Mish
from echoAI.Activation.Torch.silu import Silu
from echoAI.Activation.Torch.sine_relu import SineReLU
from echoAI.Activation.Torch.soft_clipping import SoftClipping
from echoAI.Activation.Torch.sqnl import SQNL
from echoAI.Activation.Torch.swish import Swish

# import custom activations
from echoAI.Activation.Torch.weightedTanh import WeightedTanh

sys.path.insert(0, "../")

# activation names constants
WEIGHTED_TANH = "weighted_tanh"
MISH = "mish"
SILU = "silu"
ARIA2 = "aria2"
SWISH = "swish"
ESWISH = "eswish"
BMISH = "beta_mish"
ELISH = "elish"
HELISH = "hard_elish"
MILA = "mila"
SINERELU = "sine_relu"
FTS = "fts"
SQNL = "sqnl"
ISRU = "isru"
ISRLU = "isrlu"
BENTID = "bent_id"
SC = "soft_clipping"


# create class for basic fully-connected deep neural network
class Classifier(nn.Module):
    def __init__(self, activation="weighted_tanh", inplace=False):
        super().__init__()

        # get activation the function to use
        self.activation = activation

        # set version of function to use
        self.inplace = inplace

        # initialize layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        # make sure the input tensor is flattened
        x = x.view(x.shape[0], -1)

        # apply custom activation function
        if self.activation == WEIGHTED_TANH:
            x = self.fc1(x)
            if self.inplace:
                Func.weighted_tanh(x, weight=1, inplace=self.inplace)
            else:
                x = Func.weighted_tanh(x, weight=1, inplace=self.inplace)

        if self.activation == MISH:
            x = self.fc1(x)
            if self.inplace:
                Func.mish(x, inplace=self.inplace)
            else:
                x = Func.mish(x, inplace=self.inplace)

        if self.activation == SILU:
            x = self.fc1(x)
            if self.inplace:
                Func.silu(x, inplace=self.inplace)
            else:
                x = Func.silu(x, inplace=self.inplace)

        if self.activation == ARIA2:
            x = Func.aria2(self.fc1(x))

        if self.activation == ESWISH:
            x = Func.eswish(self.fc1(x))

        if self.activation == SWISH:
            x = Func.swish(self.fc1(x))

        if self.activation == BMISH:
            x = Func.beta_mish(self.fc1(x))

        if self.activation == ELISH:
            x = Func.elish(self.fc1(x))

        if self.activation == HELISH:
            x = Func.hard_elish(self.fc1(x))

        if self.activation == MILA:
            x = Func.mila(self.fc1(x))

        if self.activation == SINERELU:
            x = Func.sineReLU(self.fc1(x))

        if self.activation == FTS:
            x = Func.fts(self.fc1(x))

        if self.activation == SQNL:
            x = Func.sqnl(self.fc1(x))

        if self.activation == ISRU:
            x = Func.isru(self.fc1(x))

        if self.activation == ISRLU:
            x = Func.isrlu(self.fc1(x))

        if self.activation == BENTID:
            x = Func.bent_id(self.fc1(x))

        if self.activation == SC:
            x = Func.soft_clipping(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


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
        default=WEIGHTED_TANH,
        help="Activation function for demonstration.",
        choices=[
            WEIGHTED_TANH,
            MISH,
            SILU,
            ARIA2,
            ESWISH,
            SWISH,
            BMISH,
            ELISH,
            HELISH,
            MILA,
            SINERELU,
            FTS,
            SQNL,
            ISRU,
            ISRLU,
            BENTID,
            SC,
        ],
    )

    # Add argument to choose the way to initialize the model
    parser.add_argument(
        "--model_initialization",
        action="store",
        default="class",
        help="Model initialization mode: use custom class or use Sequential.",
        choices=["class", "sequential"],
    )

    # Add argument to choose the way to initialize the model
    parser.add_argument(
        "--inplace",
        action="store_true",
        default=False,
        help="Use in-place of out-of-place version of activation function.",
    )

    # Parse command line arguments
    results = parser.parse_args()
    activation = results.activation
    model_initialization = results.model_initialization
    inplace = results.inplace

    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data for Fashion MNIST
    trainset = datasets.FashionMNIST(
        "~/.pytorch/F_MNIST_data/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data for Fashion MNIST
    testset = datasets.FashionMNIST(
        "~/.pytorch/F_MNIST_data/", download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    print("Create model with {activation} function.\n".format(activation=activation))

    # Initialize the model
    if model_initialization == "class":
        # Initialize the model using defined Classifier class
        model = Classifier(activation=activation, inplace=inplace)
    else:
        # Setup the activation function
        if activation == WEIGHTED_TANH:
            activation_function = WeightedTanh(weight=1, inplace=inplace)

        if activation == MISH:
            activation_function = Mish()

        if activation == SILU:
            activation_function = Silu(inplace=inplace)

        if activation == ARIA2:
            activation_function = Aria2()

        if activation == ESWISH:
            activation_function = Eswish()

        if activation == SWISH:
            activation_function = Swish()

        if activation == BMISH:
            activation_function = BetaMish()

        if activation == ELISH:
            activation_function = Elish()

        if activation == HELISH:
            activation_function = HardElish()

        if activation == MILA:
            activation_function = Mila()

        if activation == SINERELU:
            activation_function = SineReLU()

        if activation == FTS:
            activation_function = FTS()

        if activation == SQNL:
            activation_function = SQNL()

        if activation == ISRU:
            activation_function = ISRU()

        if activation == ISRLU:
            activation_function = ISRLU()

        if activation == BENTID:
            activation_function = BentID()

        if activation == SC:
            activation_function = SoftClipping()

        # Initialize the model using nn.Sequential
        model = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(784, 256)),
                    (
                        "activation",
                        activation_function,
                    ),  # use custom activation function
                    ("fc2", nn.Linear(256, 128)),
                    ("bn2", nn.BatchNorm1d(num_features=128)),
                    ("relu2", nn.ReLU()),
                    ("dropout", nn.Dropout(0.3)),
                    ("fc3", nn.Linear(128, 64)),
                    ("bn3", nn.BatchNorm1d(num_features=64)),
                    ("relu3", nn.ReLU()),
                    ("logits", nn.Linear(64, 10)),
                    ("logsoftmax", nn.LogSoftmax(dim=1)),
                ]
            )
        )

    # Train the model
    print(
        "Training the model on Fashion MNIST dataset with {} activation function.\n".format(
            activation
        )
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = 5

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")


if __name__ == "__main__":
    main()
