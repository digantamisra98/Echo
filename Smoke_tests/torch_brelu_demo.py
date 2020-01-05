"""
Script for demonstration of the BReLU activation function.
"""
# import utilities
import sys

import argparse

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# import BReLU from Echo
from echoAI.Activation.Torch.brelu import BReLU

sys.path.insert(0, "../")


class CNN(nn.Module):
    """
    Simple CNN to demonstrate BReLU activation.
    """

    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16)
        )

        self.brelu1 = BReLU.apply
        self.pool1 = nn.MaxPool2d(2)

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32)
        )

        self.brelu2 = BReLU.apply
        self.pool2 = nn.MaxPool2d(2)

        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)

        x = self.layer1(x)
        x = self.brelu1(x)
        x = self.pool1(x)

        x = self.layer2(x)
        x = self.brelu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# create class for basic fully-connected deep neural network
class Classifier(nn.Module):
    """
    Basic fully-connected network to test BReLU.
    """

    def __init__(self):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # initialize SReLU
        self.a1 = BReLU.apply
        self.a2 = BReLU.apply
        self.a3 = BReLU.apply

    def forward(self, x):
        # make sure the input tensor is flattened
        x = x.view(x.shape[0], -1)

        # apply SReLU function
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a3(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


def main():
    """
    Script for BReLU demonstration.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Argument parser")

    # Add argument to choose network architecture
    parser.add_argument(
        "--model",
        action="store",
        default="FC",
        help="Model architecture: use fully-connected model or CNN.",
        choices=["FC", "CNN"],
    )

    # Parse command line arguments
    results = parser.parse_args()
    architecture = results.model

    # apply BReLU to random tensor
    brelu_activation = BReLU.apply
    t = torch.randn((5, 5), dtype=torch.float, requires_grad=True)
    t = brelu_activation(t)

    # apply BReLU for simple model (FC or CNN depending on parameter)
    # create a model to classify Fashion MNIST dataset
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

    print("Create model with {activation} function.\n".format(activation="BReLU"))

    # create model
    if architecture == "FC":
        model = Classifier()
        criterion = nn.NLLLoss()
    else:
        model = CNN()
        criterion = nn.CrossEntropyLoss()
    print(model)

    # Train the model
    print(
        "Training the model on Fashion MNIST dataset with {} activation function.\n".format(
            "BReLU"
        )
    )

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
