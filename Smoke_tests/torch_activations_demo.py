'''
Script for demonstration of the custom activation functions
implemented in the Echo package for classification of Fashion MNIST dataset.
'''

# import basic libraries
from collections import OrderedDict
import sys
sys.path.insert(0, '../')

# import custom packages
import argparse

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# import custom activations
from Echo.Activation.Torch.weightedTanh import weightedTanh
from Echo.Activation.Torch.mish import mish
from Echo.Activation.Torch.silu import silu
from Echo.Activation.Torch.aria2 import aria2
from Echo.Activation.Torch.eswish import eswish
from Echo.Activation.Torch.swish import swish
from Echo.Activation.Torch.beta_mish import beta_mish
from Echo.Activation.Torch.elish import elish
from Echo.Activation.Torch.hard_elish import hard_elish
from Echo.Activation.Torch.mila import mila
from Echo.Activation.Torch.sine_relu import sine_relu
from Echo.Activation.Torch.fts import fts
from Echo.Activation.Torch.sqnl import sqnl
from Echo.Activation.Torch.isru import isru
from Echo.Activation.Torch.isrlu import isrlu
from Echo.Activation.Torch.bent_id import bent_id
from Echo.Activation.Torch.soft_clipping import soft_clipping
import Echo.Activation.Torch.functional as Func

# activation names constants
WEIGHTED_TANH = 'weighted_tanh'
MISH = 'mish'
SILU = 'silu'
ARIA2 = 'aria2'
SWISH = 'swish'
ESWISH = 'eswish'
BMISH = 'beta_mish'
ELISH = 'elish'
HELISH = 'hard_elish'
MILA = 'mila'
SINERELU = 'sine_relu'
FTS = 'fts'
SQNL = 'sqnl'
ISRU = 'isru'
ISRLU = 'isrlu'
BENTID = 'bent_id'
SC = 'soft_clipping'

# create class for basic fully-connected deep neural network
class Classifier(nn.Module):
    def __init__(self, activation = 'weighted_tanh', inplace = False):
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
        if (self.activation == WEIGHTED_TANH):
            x = self.fc1(x)
            if self.inplace:
                Func.weighted_tanh(x, weight = 1, inplace = self.inplace)
            else:
                x = Func.weighted_tanh(x, weight = 1, inplace = self.inplace)

        if (self.activation == MISH):
            x = self.fc1(x)
            if self.inplace:
                Func.mish(x, inplace = self.inplace)
            else:
                x = Func.mish(x, inplace = self.inplace)

        if (self.activation == SILU):
            x = self.fc1(x)
            if self.inplace:
                Func.silu(x, inplace = self.inplace)
            else:
                x = Func.silu(x, inplace = self.inplace)

        if (self.activation == ARIA2):
            x = Func.aria2(self.fc1(x))

        if (self.activation == ESWISH):
            x = Func.eswish(self.fc1(x))

        if (self.activation == SWISH):
            x = Func.swish(self.fc1(x))

        if (self.activation == BMISH):
            x = Func.beta_mish(self.fc1(x))

        if (self.activation == ELISH):
            x = Func.elish(self.fc1(x))

        if (self.activation == HELISH):
            x = Func.hard_elish(self.fc1(x))

        if (self.activation == MILA):
            x = Func.mila(self.fc1(x))

        if (self.activation == SINERELU):
            x = Func.sineReLU(self.fc1(x))

        if (self.activation == FTS):
            x = Func.fts(self.fc1(x))

        if (self.activation == SQNL):
            x = Func.sqnl(self.fc1(x))

        if (self.activation == ISRU):
            x = Func.isru(self.fc1(x))

        if (self.activation == ISRLU):
            x = Func.isrlu(self.fc1(x))

        if (self.activation == BENTID):
            x = Func.bent_id(self.fc1(x))

        if (self.activation == SC):
            x = Func.soft_clipping(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

def main():
    '''
    Demonstrate custom activation functions to classify Fashion MNIST
    '''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    # Add argument to choose one of the activation functions
    parser.add_argument('--activation', action='store', default = WEIGHTED_TANH,
                        help='Activation function for demonstration.',
                        choices = [WEIGHTED_TANH, MISH, SILU, ARIA2, ESWISH, SWISH, BMISH, ELISH, HELISH, MILA, SINERELU, FTS, SQNL, ISRU, ISRLU, BENTID, SC])

    # Add argument to choose the way to initialize the model
    parser.add_argument('--model_initialization', action='store', default = 'class',
                        help='Model initialization mode: use custom class or use Sequential.',
                        choices = ['class', 'sequential'])

    # Add argument to choose the way to initialize the model
    parser.add_argument('--inplace', action="store_true", default = False,
                        help='Use in-place of out-of-place version of activation function.')

    # Parse command line arguments
    results = parser.parse_args()
    activation = results.activation
    model_initialization = results.model_initialization
    inplace = results.inplace

    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data for Fashion MNIST
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data for Fashion MNIST
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    print("Create model with {activation} function.\n".format(activation = activation))

    # Initialize the model
    if (model_initialization == 'class'):
        # Initialize the model using defined Classifier class
        model = Classifier(activation = activation, inplace = inplace)
    else:
        # Setup the activation function
        if (activation == WEIGHTED_TANH):
            activation_function = weightedTanh(weight = 1, inplace = inplace)

        if (activation == MISH):
            activation_function = mish(inplace = inplace)

        if (activation == SILU):
            activation_function = silu(inplace = inplace)

        if (activation == ARIA2):
            activation_function = aria2()

        if (activation == ESWISH):
            activation_function = eswish()

        if (activation == SWISH):
            activation_function = swish()

        if (activation == BMISH):
            activation_function = beta_mish()

        if (activation == ELISH):
            activation_function = elish()

        if (activation == HELISH):
            activation_function = hard_elish()

        if (activation == MILA):
            activation_function = mila()

        if (activation == SINERELU):
            activation_function = sine_relu()

        if (activation == FTS):
            activation_function = fts()

        if (activation == SQNL):
            activation_function = sqnl()

        if (activation == ISRU):
            activation_function = isru()

        if (activation == ISRLU):
            activation_function = isrlu()

        if (activation == BENTID):
            activation_function = bent_id()

        if (activation == SC):
            activation_function = soft_clipping()

        # Initialize the model using nn.Sequential
        model = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(784, 256)),
                              ('activation', activation_function), # use custom activation function
                              ('fc2', nn.Linear(256, 128)),
                              ('bn2', nn.BatchNorm1d(num_features=128)),
                              ('relu2', nn.ReLU()),
                              ('dropout', nn.Dropout(0.3)),
                              ('fc3', nn.Linear(128, 64)),
                              ('bn3', nn.BatchNorm1d(num_features=64)),
                              ('relu3', nn.ReLU()),
                              ('logits', nn.Linear(64, 10)),
                              ('logsoftmax', nn.LogSoftmax(dim=1))]))

    # Train the model
    print("Training the model on Fashion MNIST dataset with {} activation function.\n".format(activation))

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

if __name__ == '__main__':
    main()
