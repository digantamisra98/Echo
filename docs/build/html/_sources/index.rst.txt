.. Echo documentation master file, created on Tue Jun 11 10:55:29 2019.

################################
Welcome to Echo AI documentation!
################################

.. toctree::
   :maxdepth: 5
   :caption: Contents:


.. contents:: Table of Contents
 :local:
 :depth: 1


About
================================

**EchoAI Package** is created to provide an implementation of the most promising mathematical algorithms, which are missing in the most popular deep learning libraries, such as `PyTorch <https://pytorch.org/>`_, `Keras <https://keras.io/>`_ and
`TensorFlow <https://www.tensorflow.org/>`_.

Implemented Activation Functions
--------------------------------

The list of activation functions implemented in Echo:

====   ====================   =========================================   ==============================================================   ================
#      Activation             PyTorch                                     Keras                                                            TensorFlow Keras
====   ====================   =========================================   ==============================================================   ================
1      Weighted Tanh          :ref:`Torch.WeightedTanh`                   :ref:`Keras.weighted_tanh`                                       -
2      Aria2                  :ref:`Torch.aria2`                          :ref:`Keras.aria2`                                               -
3      SiLU                   :ref:`Torch.silu`                           -                                                                -
4      E-Swish                :ref:`Torch.eswish`                         :ref:`Keras.eswish`                                              -
5      Swish                  :ref:`Torch.swish`                          :ref:`Keras.swish`                                               -
6      ELiSH                  :ref:`Torch.elish`                          :ref:`Keras.elish`                                               -
7      Hard ELiSH             :ref:`Torch.hard_elish`                     :ref:`Keras.hard_elish`                                          -
8      Mila                   :ref:`Torch.mila`                           :ref:`Keras.mila`                                                -
9      SineReLU               :ref:`Torch.sine_relu`                      :ref:`Keras.sineReLU`                                            -
10     Flatten T-Swish        :ref:`Torch.fts`                            :ref:`Keras.fts`                                                 -
11     SQNL                   :ref:`Torch.sqnl`                           :ref:`Keras.sqnl`                                                -
12     Mish                   :ref:`Torch.mish`                           :ref:`Keras.mish`                                                -
13     Beta Mish              :ref:`Torch.beta_mish`                      :ref:`Keras.beta_mish`                                           -
14     ISRU                   :ref:`Torch.isru`                           :ref:`Keras.isru`                                                -
15     ISRLU                  :ref:`Torch.isrlu`                          :ref:`Keras.isrlu`                                               -
16     Bent's Identity        :ref:`Torch.bent_id`                        :ref:`Keras.bent_id`                                             -
17     Soft Clipping          :ref:`Torch.soft_clipping`                  :ref:`Keras.soft_clipping`                                       -
18     SReLU                  :ref:`Torch.srelu`                          :ref:`Keras.srelu`                                               -
19     BReLU                  :ref:`Torch.brelu`                          -                                                                -
20     APL                    :ref:`Torch.apl`                            -                                                                -
21     Soft Exponential       :ref:`Torch.soft_exponential`               :ref:`Keras.soft_exponential`                                    -
22     Maxout                 :ref:`Torch.maxout`                         -                                                                -
23     CELU                   -                                           :ref:`Keras.celu`                                                -
23     ReLU6                  -                                           :ref:`Keras.relu6`                                               -
24     Hard Tanh              -                                           :ref:`Keras.hard_tanh`                                           -
25     Log Sigmoid            -                                           :ref:`Keras.log_sigmoid`                                         -
26     Tanh Shrink            -                                           :ref:`Keras.tanh_shrink`                                         -
27     Hard Shrink            -                                           :ref:`Keras.hard_shrink`                                         -
28     Soft Shrink            -                                           :ref:`Keras.soft_shrink`                                         -
29     Softmin                -                                           :ref:`Keras.softmin`                                             -
30     LogSoftmax             -                                           :ref:`Keras.log_softmax`                                         -
====   ====================   =========================================   ==============================================================   ================

Installation
================================
To install EchoAI package from source follow the instructions below:

1. Clone or download `GitHub repository <https://github.com/digantamisra98/Echo>`_.

2. Navigate to **echoAI** folder:
  >>> $ cd Echo

3. Install the package with pip:
  >>> $ pip install .

To install EchoAI package from PyPI follow the instructions below:

>>> $ pip install echoAI

Examples
================================

Torch Activation Functions
--------------------------------

The following code block contains an example of usage of a PyTorch activation function
from Echo package:

.. code-block:: python
   :emphasize-lines: 2,3,21,37

   # import activations from EchoAI
   from echoAI.Activation.Torch.weightedTanh import WeightedTanh
   import echoAI.Activation.Torch.functional as Func

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
                            ('wtahn1',  WeightedTanh(weight = 1)),
                            ('fc2', nn.Linear(256, 128)),
                            ('bn2', nn.BatchNorm1d(num_features=128)),
                            ('relu2', nn.ReLU()),
                            ('dropout', nn.Dropout(0.3)),
                            ('fc3', nn.Linear(128, 64)),
                            ('bn3', nn.BatchNorm1d(num_features=64)),
                            ('relu3', nn.ReLU()),
                            ('logits', nn.Linear(64, 10)),
                            ('logsoftmax', nn.LogSoftmax(dim=1))]))

Keras Activation Functions
--------------------------------

The following code block contains an example of usage of a Keras activation function
from Echo package:

.. code-block:: python
   :emphasize-lines: 2,27

   # Import the activation function from Echo package
   from echoAI.Activation.Keras.custom_activations import WeightedTanh

   # Define the CNN model
   def CNNModel(input_shape):
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

       # CONV -> BN -> Activation Block applied to X
       X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0')(X)
       X = BatchNormalization(axis = 3, name = 'bn0')(X)

       # Use custom activation function from Echo package
       X = WeightedTanh()(X)

       # MAXPOOL
       X = MaxPooling2D((2, 2), name='max_pool')(X)

       # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
       X = Flatten()(X)
       X = Dense(10, activation='softmax', name='fc')(X)

       # Create model
       model = Model(inputs = X_input, outputs = X, name='CNNModel')

       return model

   # Create the model
   model = CNNModel((28,28,1))

   # Compile the model
   model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = ["accuracy"])


PyTorch Extensions
================================

.. _Torch.aria2:

Torch.aria2
--------------------------------
.. automodule:: echoAI.Activation.Torch.aria2
  :members:

.. _Torch.mish:

Torch.Mish
--------------------------------
.. automodule:: echoAI.Activation.Torch.mish
  :members:

.. _Torch.beta_mish:

Torch.BetaMish
--------------------------------
.. automodule:: echoAI.Activation.Torch.beta_mish
  :members:

.. _Torch.silu:

Torch.Silu
--------------------------------
.. automodule:: echoAI.Activation.Torch.silu
  :members:

.. _Torch.eswish:

Torch.Eswish
--------------------------------
.. automodule:: echoAI.Activation.Torch.eswish
  :members:

.. _Torch.swish:

Torch.Swish
--------------------------------
.. automodule:: echoAI.Activation.Torch.swish
  :members:

.. _Torch.elish:

Torch.Elish
--------------------------------
.. automodule:: echoAI.Activation.Torch.elish
  :members:

.. _Torch.hard_elish:

Torch.HardElish
--------------------------------
.. automodule:: echoAI.Activation.Torch.hard_elish
  :members:

.. _Torch.mila:

Torch.Mila
--------------------------------
.. automodule:: echoAI.Activation.Torch.mila
  :members:

.. _Torch.sine_relu:

Torch.SineReLU
--------------------------------
.. automodule:: echoAI.Activation.Torch.sine_relu
  :members:

.. _Torch.fts:

Torch.FTS
--------------------------------
.. automodule:: echoAI.Activation.Torch.fts
  :members:

.. _Torch.sqnl:

Torch.SQNL
--------------------------------
.. automodule:: echoAI.Activation.Torch.sqnl
  :members:

.. _Torch.isru:

Torch.ISRU
--------------------------------
.. automodule:: echoAI.Activation.Torch.isru
  :members:

.. _Torch.isrlu:

Torch.ISRLU
--------------------------------
.. automodule:: echoAI.Activation.Torch.isrlu
  :members:

.. _Torch.bent_id:

Torch.BentID
--------------------------------
.. automodule:: echoAI.Activation.Torch.bent_id
  :members:

.. _Torch.soft_clipping:

Torch.SoftClipping
--------------------------------
.. automodule:: echoAI.Activation.Torch.soft_clipping
  :members:

.. _Torch.WeightedTanh:

Torch.WeightedTanh
--------------------------------
.. automodule:: echoAI.Activation.Torch.weightedTanh
  :members:

.. _Torch.srelu:

Torch.SReLU
--------------------------------
.. automodule:: echoAI.Activation.Torch.srelu
  :members:

.. _Torch.brelu:

Torch.BReLU
--------------------------------
.. automodule:: echoAI.Activation.Torch.brelu
  :members:

.. _Torch.apl:

Torch.APL
--------------------------------
.. automodule:: echoAI.Activation.Torch.apl
  :members:

.. _Torch.soft_exponential:

Torch.SoftExponential
--------------------------------
.. automodule:: echoAI.Activation.Torch.soft_exponential
  :members:

.. _Torch.maxout:

Torch.Maxout
--------------------------------
.. automodule:: echoAI.Activation.Torch.maxout
  :members:

echoAI.Activation.Torch.functional
--------------------------------
.. automodule:: echoAI.Activation.Torch.functional
   :members:

Keras Extensions
================================

.. _Keras.mish:

Keras.Mish
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Mish
  :members:

.. _Keras.weighted_tanh:

Keras.WeightedTanh
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.WeightedTanh
  :members:

.. _Keras.aria2:

Keras.Aria2
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Aria2
  :members:

.. _Keras.eswish:

Keras.Eswish
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Eswish
  :members:

.. _Keras.swish:

Keras.Swish
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Swish
  :members:

.. _Keras.elish:

Keras.Elish
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Elish
  :members:

.. _Keras.hard_elish:

Keras.HardElish
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.HardElish
  :members:

.. _Keras.mila:

Keras.Mila
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Mila
  :members:

.. _Keras.sineReLU:

Keras.SineReLU
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SineReLU
  :members:

.. _Keras.fts:

Keras.FTS
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.FTS
  :members:

.. _Keras.sqnl:

Keras.SQNL
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SQNL
  :members:

.. _Keras.beta_mish:

Keras.BetaMish
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.BetaMish
  :members:

.. _Keras.isru:

Keras.ISRU
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.ISRU
  :members:

.. _Keras.isrlu:

Keras.ISRLU
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.ISRLU
  :members:

.. _Keras.bent_id:

Keras.BentID
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.BentID
  :members:

.. _Keras.soft_clipping:

Keras.SoftClipping
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SoftClipping
  :members:

.. _Keras.celu:

Keras.Celu
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.Celu
  :members:

.. _Keras.relu6:

Keras.ReLU6
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.ReLU6
  :members:

.. _Keras.hard_tanh:

Keras.HardTanh
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.HardTanh
  :members:

.. _Keras.log_sigmoid:

Keras.LogSigmoid
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.LogSigmoid
  :members:

.. _Keras.tanh_shrink:

Keras.TanhShrink
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.TanhShrink
  :members:

.. _Keras.hard_shrink:

Keras.HardShrink
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.HardShrink
  :members:

.. _Keras.soft_shrink:

Keras.SoftShrink
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SoftShrink
  :members:

.. _Keras.softmin:

Keras.SoftMin
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SoftMin
  :members:

.. _Keras.log_softmax:

Keras.LogSoftmax
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.LogSoftmax
  :members:

.. _Keras.soft_exponential:

Keras.SoftExponential
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SoftExponential
  :members:

.. _Keras.srelu:

Keras.SReLU
--------------------------------
.. automodule:: echoAI.Activation.Keras.custom_activations.SReLU
  :members:

Indices and tables
================================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
