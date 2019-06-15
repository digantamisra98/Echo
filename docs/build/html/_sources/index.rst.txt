.. Echo documentation master file, created by
   sphinx-quickstart on Tue Jun 11 10:55:29 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################################
Welcome to Echo's documentation!
################################

.. toctree::
   :maxdepth: 5
   :caption: Contents:


.. contents:: Table of Contents
 :local:


About
================================

**Echo Package** is created to provide an implementation of the most promising mathematical algorithms, which are missing in the most popular deep learning libraries, such as `PyTorch <https://pytorch.org/>`_, `Keras <https://keras.io/>`_ and
`TensorFlow <https://www.tensorflow.org/>`_.

Implemented Activation Functions
--------------------------------

List of activation functions implemented in Echo:

1. PyTorch:
  * Weighted Tanh (see :mod:`Echo.Activation.Torch.weightedTanh`)
  * Aria2 (see :mod:`Echo.Activation.Torch.aria2`)
  * Swish (see :mod:`Echo.Activation.Torch.swish`)
  * E-Swish (see :mod:`Echo.Activation.Torch.eswish`)
  * SwishX (see :mod:`Echo.Activation.Torch.swishx`)
  * Mish (see :mod:`Echo.Activation.Torch.mish`)
  * Beta Mish (see :mod:`Echo.Activation.Torch.beta_mish`)

Installation
================================
To install Echo package follow the instructions below:

1. Clone or download `GitHub repository <https://github.com/digantamisra98/Echo>`_.

2. Navigate to **Echo** folder:
  >>> $ cd Echo

3. Install the package with pip:
  >>> $ pip install .

Torch Examples
================================

Activation Functions
--------------------------------

The following code block contains an example of usage of an activation function
from Echo package:

.. code-block:: python
   :emphasize-lines: 2,3,21,37

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

Echo API Reference
================================

Echo.Activation.Torch.aria2
--------------------------------
.. automodule:: Echo.Activation.Torch.aria2
  :members:

Echo.Activation.Torch.mish
--------------------------------
.. automodule:: Echo.Activation.Torch.mish
  :members:

Echo.Activation.Torch.beta_mish
--------------------------------
.. automodule:: Echo.Activation.Torch.beta_mish
  :members:

Echo.Activation.Torch.swish
--------------------------------
.. automodule:: Echo.Activation.Torch.swish
  :members:

Echo.Activation.Torch.eswish
--------------------------------
.. automodule:: Echo.Activation.Torch.eswish
  :members:

Echo.Activation.Torch.swishx
--------------------------------
.. automodule:: Echo.Activation.Torch.swishx
  :members:

Echo.Activation.Torch.weightedTanh
--------------------------------
.. automodule:: Echo.Activation.Torch.weightedTanh
  :members:

Echo.Activation.Torch.functional
--------------------------------
.. automodule:: Echo.Activation.Torch.functional
   :members:

Indices and tables
--------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
