'''
Script for demonstration of the BReLU activation function.
'''
# import utilities
import sys
sys.path.insert(0, '../')
import argparse

# import pytorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# import Maxout from Echo
from Echo.Activation.Torch.maxout import maxout

#This is an example for image reconstruction but you can modify it as you want.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.mo1=maxout.apply
        self.layer2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.mo2 = maxout.apply
        self.layer3 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.mo3 = maxout.apply #max_out on line 8 if class Maxout is 4, it will output 1 feature map here
        self.fc = nn.Linear(1*28*28, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        out = self.mo1(self.layer1(x))
        out = self.mo2(self.layer2(out))
        out = self.mo3(self.layer3(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def main():
    '''
    Script for Maxout demonstration.
    '''
    # apply Maxout for simple model (FC or CNN depending on parameter)
    # create a model to classify Fashion MNIST dataset
    # Define a transform
    transform = transforms.Compose([transforms.ToTensor()])

    # Download and load the training data for Fashion MNIST
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data for Fashion MNIST
    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    print("Create model with {activation} function.\n".format(activation = 'Maxout'))

    # create model
    model = CNN()
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Training the model on Fashion MNIST dataset with {} activation function.\n".format('Maxout'))

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
