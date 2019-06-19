<p align="left">
  <img width="270" src="Observations/logo_transparent.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

# Echo
Python package containing all mathematical backend algorithms used in Machine Learning.

## Table of Contents
* [About the Project](#about)
  * [Activation Functions](#activation-functions)
* [Repository Structure](#repository-structure)
* [Setup Instructions](#setup-instructions)
* [Code Examples](#code-examples)
  * [PyTorch Activation Functions](#pytorch-activation-functions)

## About
**Echo Package** is created to provide an implementation of the most promising mathematical algorithms, which are missing in the most popular deep learning libraries, such as [PyTorch](https://pytorch.org/), [Keras](https://keras.io/) and
[TensorFlow](https://www.tensorflow.org/).

### Activation Functions
The package contains implementation for following activation functions:

* Weighted Tanh

![equation](https://latex.codecogs.com/gif.latex?weightedtanh%28x%29%20%3D%20tanh%28x%20*%20weight%29)

* [Swish](https://arxiv.org/pdf/1710.05941.pdf)

![equation](https://latex.codecogs.com/gif.latex?Swish%28x%29%20%3D%20x%20*%20sigmoid%28x%29)

* [ESwish](https://arxiv.org/abs/1801.07145)

![equation](https://latex.codecogs.com/gif.latex?ESwish%28x%2C%20%5Cbeta%29%20%3D%20%5Cbeta*x*sigmoid%28x%29)

* [SwishX](https://arxiv.org/pdf/1710.05941.pdf)

![equation](https://latex.codecogs.com/gif.latex?SwishX%28x%2C%20%5Cbeta%29%20%3D%20x*sigmoid%28%5Cbeta*x%29%20%3D%20%5Cfrac%7Bx%7D%7B%281&plus;e%5E%7B-%5Cbeta*x%7D%29%7D)

* [Aria2](https://arxiv.org/abs/1805.08878)

![equation](https://latex.codecogs.com/gif.latex?Aria2%28x%2C%20%5Calpha%2C%20%5Cbeta%29%20%3D%20%281&plus;e%5E%7B-%5Cbeta*x%7D%29%5E%7B-%5Calpha%7D)

* [ELiSH](https://arxiv.org/pdf/1808.00783.pdf)

![equation](https://latex.codecogs.com/gif.latex?ELiSH%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20/%20%281&plus;e%5E%7B-x%7D%29%2C%20x%20%5Cgeq%200%20%5C%5C%20%28e%5E%7Bx%7D%20-%201%29%20/%20%281%20&plus;%20e%5E%7B-x%7D%29%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)

* [HardELiSH](https://arxiv.org/pdf/1808.00783.pdf)

![equation](https://latex.codecogs.com/gif.latex?HardELiSH%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20%5Ctimes%20max%280%2C%20min%281%2C%20%28x%20&plus;%201%29%20/%202%29%29%2C%20x%20%5Cgeq%200%20%5C%5C%20%28e%5E%7Bx%7D%20-%201%29%5Ctimes%20max%280%2C%20min%281%2C%20%28x%20&plus;%201%29%20/%202%29%29%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)

* Mila

![equation](https://latex.codecogs.com/gif.latex?mila%28x%29%20%3D%20x%20*%20tanh%28ln%281%20&plus;%20e%5E%7B%5Cbeta%20&plus;%20x%7D%29%29%20%3D%20x%20*%20tanh%28softplus%28%5Cbeta%20&plus;%20x%29%29)

* [SineReLU](https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d)

![equation](https://latex.codecogs.com/gif.latex?SineReLU%28x%2C%20%5Cepsilon%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20%2C%20x%20%3E%200%20%5C%5C%20%5Cepsilon%20*%20%28sin%28x%29%20-%20cos%28x%29%29%2C%20x%20%5Cleq%200%20%5Cend%7Bmatrix%7D%5Cright)

* [Flatten T-Swish](https://arxiv.org/pdf/1812.06247.pdf)

![equation](https://latex.codecogs.com/gif.latex?FTS%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Cfrac%7Bx%7D%7B1%20&plus;%20e%5E%7B-x%7D%7D%20%2C%20x%20%5Cgeq%200%20%5C%5C%200%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)

* [SQNL](https://ieeexplore.ieee.org/document/8489043)

![equation](https://latex.codecogs.com/gif.latex?FTS%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%2C%20x%20%3E%202%20%5C%5C%20x%20-%20%5Cfrac%7Bx%5E2%7D%7B4%7D%2C%200%20%5Cleq%20x%20%5Cleq%202%20%5C%5C%20x%20&plus;%20%5Cfrac%7Bx%5E2%7D%7B4%7D%2C%20-2%20%5Cleq%20x%20%3C%200%20%5C%5C%20-1%2C%20x%20%3C%20-2%20%5Cend%7Bmatrix%7D%5Cright.)

* Mish
* Beta Mish

## Repository Structure
The repository has the following structure:
```python
- Echo # main package directory
| - Activation # sub-package containing activation functions implementation
| |- Torch  # sub-package containing implementation for PyTorch
| |- __init__.py  # classification result page of web app
| | | - functional.py # script which contains implementation of activation functions
| | | - weightedTanh.py # activation functions wrapper class for PyTorch
| | | - ... # PyTorch activation functions wrappers
| - __init__.py

- Observations # Folder containing other assets

- docs # Sphinx documentation folder

- LICENSE # license file
- README.md
- setup.py # package setup file
- torch_activations_demo.py # script, which contains the demonstration of PyTorch activations usage
```

## Setup Instructions
To install Echo package follow the instructions below:

1. Clone or download [GitHub repository](https://github.com/digantamisra98/Echo).

2. Navigate to **Echo** folder:
  
  ```$ cd Echo```

3. Install the package with pip:
  
  ```$ pip install . ```

## Code Examples

### PyTorch Activation Functions

The following code block contains an example of usage of a PyTorch activation function
from Echo package:

```python
   # import activations from Echo
   from Echo.Activation.Torch.weightedTanh import weightedTanh
   import Echo.Activation.Torch.functional as Func

   # use activations in layers of model defined in class
   class Classifier(nn.Module):
       def __init__(self):
           super().__init__()

           # initialize layers
           self.fc1 = nn.Linear(784, 256)
           self.fc2 = nn.Linear(256, 128)
           self.fc3 = nn.Linear(128, 64)
           self.fc4 = nn.Linear(64, 10)

       def forward(self, x):
           # make sure the input tensor is flattened
           x = x.view(x.shape[0], -1)

           # apply activation function from Echo
           x = Func.weighted_tanh(self.fc1(x), weight = 1)

           x = F.relu(self.fc2(x))
           x = F.relu(self.fc3(x))
           x = F.log_softmax(self.fc4(x), dim=1)

           return x

   def main():
       # Initialize the model using defined Classifier class
       model = Classifier()

       # Create model with Sequential
       model = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(784, 256)),
                            # use activation function from Echo
                            ('wtahn1',  weightedTanh(weight = 1)),
                            ('fc2', nn.Linear(256, 128)),
                            ('bn2', nn.BatchNorm1d(num_features=128)),
                            ('relu2', nn.ReLU()),
                            ('dropout', nn.Dropout(0.3)),
                            ('fc3', nn.Linear(128, 64)),
                            ('bn3', nn.BatchNorm1d(num_features=64)),
                            ('relu3', nn.ReLU()),
                            ('logits', nn.Linear(64, 10)),
                            ('logsoftmax', nn.LogSoftmax(dim=1))]))
```
The following script contains a comprehensive demonstration of usage of PyTorch activation functions: [torch_activations_demo.py](https://github.com/digantamisra98/Echo/blob/Dev-adeis/torch_activations_demo.py)
