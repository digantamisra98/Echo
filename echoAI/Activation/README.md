### Activation Functions
The package contains implementation for following activation functions (✅ - implemented functions, 🕑 - functions to be implemented soon, :white_large_square: - function is implemented in the original deep learning package):

|#| Function | Equation  | PyTorch | TensorFlow |
| --- | --- | --- | --- | --- | 
|1| [Mish](https://www.bmvc2020-conference.com/assets/papers/0928.pdf) | ![equation](https://latex.codecogs.com/gif.latex?mish%28x%29%20%3D%20x%20*%20tanh%28ln%281%20&plus;%20e%5Ex%29%29) | ✅ | ✅ |
|2| [Swish](https://arxiv.org/pdf/1710.05941.pdf) | ![equation](https://latex.codecogs.com/gif.latex?SwishX%28x%2C%20%5Cbeta%29%20%3D%20x*sigmoid%28%5Cbeta*x%29%20%3D%20%5Cfrac%7Bx%7D%7B%281&plus;e%5E%7B-%5Cbeta*x%7D%29%7D)  | ✅ | ✅  |
|3| [ESwish](https://arxiv.org/abs/1801.07145) | ![equation](https://latex.codecogs.com/gif.latex?ESwish%28x%2C%20%5Cbeta%29%20%3D%20%5Cbeta*x*sigmoid%28x%29) | ✅| ✅  |
|4| [Aria2](https://arxiv.org/abs/1805.08878) | ![equation](https://latex.codecogs.com/gif.latex?Aria2%28x%2C%20%5Calpha%2C%20%5Cbeta%29%20%3D%20%281&plus;e%5E%7B-%5Cbeta*x%7D%29%5E%7B-%5Calpha%7D) | ✅ | ✅  |
|5| [ELiSH](https://arxiv.org/pdf/1808.00783.pdf) | ![equation](https://latex.codecogs.com/gif.latex?ELiSH%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20/%20%281&plus;e%5E%7B-x%7D%29%2C%20x%20%5Cgeq%200%20%5C%5C%20%28e%5E%7Bx%7D%20-%201%29%20/%20%281%20&plus;%20e%5E%7B-x%7D%29%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)  | ✅ | ✅  |
|6| [HardELiSH](https://arxiv.org/pdf/1808.00783.pdf) | ![equation](https://latex.codecogs.com/gif.latex?HardELiSH%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20%5Ctimes%20max%280%2C%20min%281%2C%20%28x%20&plus;%201%29%20/%202%29%29%2C%20x%20%5Cgeq%200%20%5C%5C%20%28e%5E%7Bx%7D%20-%201%29%5Ctimes%20max%280%2C%20min%281%2C%20%28x%20&plus;%201%29%20/%202%29%29%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)  | ✅ | ✅  |
|7| [Flatten T-Swish](https://arxiv.org/pdf/1812.06247.pdf) | ![equation](https://latex.codecogs.com/gif.latex?FTS%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20%5Cfrac%7Bx%7D%7B1%20&plus;%20e%5E%7B-x%7D%7D%20%2C%20x%20%5Cgeq%200%20%5C%5C%200%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)  | ✅ | ✅  |
|8| [SQNL](https://ieeexplore.ieee.org/document/8489043) | ![equation](https://latex.codecogs.com/gif.latex?SQNL%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%2C%20x%20%3E%202%20%5C%5C%20x%20-%20%5Cfrac%7Bx%5E2%7D%7B4%7D%2C%200%20%5Cleq%20x%20%5Cleq%202%20%5C%5C%20x%20&plus;%20%5Cfrac%7Bx%5E2%7D%7B4%7D%2C%20-2%20%5Cleq%20x%20%3C%200%20%5C%5C%20-1%2C%20x%20%3C%20-2%20%5Cend%7Bmatrix%7D%5Cright.)  | ✅ | ✅  |
|9| [ISRU](https://arxiv.org/pdf/1710.09967.pdf) | ![equation](https://latex.codecogs.com/gif.latex?ISRU%28x%29%20%3D%20%5Cfrac%7Bx%7D%7B%5Csqrt%7B1%20&plus;%20%5Calpha%20*%20x%5E2%7D%7D) | ✅ | ✅ |
|10| [ISRLU](https://arxiv.org/pdf/1710.09967.pdf) | ![equation](https://latex.codecogs.com/gif.latex?ISRLU%28x%29%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%2C%20x%5Cgeq%200%20%5C%5C%20x%20*%20%28%5Cfrac%7B1%7D%7B%5Csqrt%7B1%20&plus;%20%5Calpha*x%5E2%7D%7D%29%2C%20x%20%3C0%20%5Cend%7Bmatrix%7D%5Cright.) | ✅ | ✅  |
|11| [Soft Clipping](https://arxiv.org/pdf/1810.11509.pdf) | ![equation](https://latex.codecogs.com/gif.latex?SC%28x%29%20%3D%201%20/%20%5Calpha%20*%20log%28%5Cfrac%7B1%20&plus;%20e%5E%7B%5Calpha%20*%20x%7D%7D%7B1%20&plus;%20e%5E%7B%5Calpha%20*%20%28x-1%29%7D%7D%29)  | ✅ | ✅ |
|12| [SReLU](https://arxiv.org/pdf/1512.07030.pdf) | ![equation](https://latex.codecogs.com/gif.latex?SReLU%28x_i%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20t_i%5Er%20&plus;%20a_i%5Er%28x_i%20-%20t_i%5Er%29%2C%20x_i%20%5Cgeq%20t_i%5Er%20%5C%5C%20x_i%2C%20t_i%5Er%20%3E%20x_i%20%3E%20t_i%5El%5C%5C%20t_i%5El%20&plus;%20a_i%5El%28x_i%20-%20t_i%5El%29%2C%20x_i%20%5Cleq%20t_i%5El%20%5C%5C%20%5Cend%7Bmatrix%7D%5Cright.) | ✅ | ✅  |
|13| [BReLU](https://arxiv.org/pdf/1709.04054.pdf) | ![equation](https://latex.codecogs.com/gif.latex?BReLU%28x_i%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20f%28x_i%29%2C%20i%20%5Cmod%202%20%3D%200%5C%5C%20-%20f%28-x_i%29%2C%20i%20%5Cmod%202%20%5Cneq%200%20%5Cend%7Bmatrix%7D%5Cright.) | ✅ | ✅  |
|14| [APL](https://arxiv.org/pdf/1412.6830.pdf) | ![equation](https://latex.codecogs.com/gif.latex?APL%28x_i%29%20%3D%20max%280%2Cx%29%20&plus;%20%5Csum_%7Bs%3D1%7D%5E%7BS%7D%7Ba_i%5Es%20*%20max%280%2C%20-x%20&plus;%20b_i%5Es%29%7D)  | ✅ | ✅ |
|15| [Soft Exponential](https://arxiv.org/pdf/1602.01321.pdf) | ![equation](https://latex.codecogs.com/gif.latex?SoftExponential%28x%2C%20%5Calpha%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20-%20%5Cfrac%7Blog%281%20-%20%5Calpha%28x%20&plus;%20%5Calpha%29%29%7D%7B%5Calpha%7D%2C%20%5Calpha%20%3C%200%5C%5C%20x%2C%20%5Calpha%20%3D%200%5C%5C%20%5Cfrac%7Be%5E%7B%5Calpha%20*%20x%7D%20-%201%7D%7B%5Calpha%7D%20&plus;%20%5Calpha%2C%20%5Calpha%20%3E%200%20%5Cend%7Bmatrix%7D%5Cright.) | ✅ | ✅ |
|16| [Maxout](https://arxiv.org/pdf/1302.4389.pdf) | ![equation](https://latex.codecogs.com/gif.latex?maxout%28%5Cvec%7Bx%7D%29%20%3D%20max_i%28x_i%29)| ✅ | ✅ |
|17| [RReLU](https://arxiv.org/pdf/1505.00853.pdf) | ![equation](https://latex.codecogs.com/gif.latex?RReLU_j_i%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x_j_i%20%2C%5Cif%20%5C%20x_j_i%5Cgeq%200%5C%5C%20a_j_i%20*%20x_j_i%2C%20%5Cif%20%5C%20x_j_i%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.%20a_j_i%20%5Csim%20U%28l%2C%20u%29%2C%20%5Cl%3C%20u%20%5Cand%20%5C%20l%2C%20u%20%5Cin%20%5B0%2C1%29) | ⬜ | 🕑 |
|18| [CELU](https://arxiv.org/pdf/1704.07483.pdf) | ![equation](https://latex.codecogs.com/gif.latex?CELU%28x%2C%20%5Calpha%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20x%20%2C%20x%20%5Cgeq%200%5C%5C%20%5Calpha%20*%20%28e%5E%7B%5Cfrac%7Bx%7D%7B%5Calpha%7D%7D%20-%201%29%2C%20x%20%3C%200%20%5Cend%7Bmatrix%7D%5Cright.)  | ⬜ | ✅ |
|19| [GLU](https://arxiv.org/pdf/1612.08083.pdf) | ![equation](https://latex.codecogs.com/gif.latex?GLU%28a%2Cb%29%3Da%20%5Cotimes%20%5Csigma%28b%29%2C%20%5Cotimes%20-%20element%20%5C%20wise%20%5C%20product) | ⬜ | 🕑 | 🕑|
|20| [SiLU](https://arxiv.org/abs/1702.03118) | | ✅ | ✅ |🕑 |
|21| [GELU](https://arxiv.org/abs/1606.08415) | | 🕑| ✅ | 🕑|
|22| [CReLU](https://arxiv.org/abs/1603.05201)| | 🕑| ✅ | 🕑|
|23| [ProbAct](https://arxiv.org/abs/1905.10761)| | 🕑| 🕑 | 🕑|
|24| [Noisy Activation Function](https://arxiv.org/pdf/1603.00391.pdf) | | 🕑| 🕑 | 🕑|
|25| [NLReLU](https://arxiv.org/abs/1908.03682) |  |✅ | ✅ |🕑 |

### Code Examples:

Sample scripts are provided in [Smoke_tests](/Smoke_tests) folder.

__PyTorch__:

```python
# import PyTorch
import torch

# import activation function from echoAI
from echoAI.Activation.Torch.mish import Mish

# apply activation function
mish = Mish()
t = torch.tensor(0.1)
t_mish = mish(t)

```

__TensorFlow Keras__:

```python
#import tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten

# import activation function from echoAI
from echoAI.Activation.TF_Keras.custom_activation import Mish

model = tf.keras.Sequential([
    layers.Flatten(),
    layers.Dense(128, input_shape=(784,)),
    Mish(), # use the activation function
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')])

# Compile the model
model.compile(optimizer = "adam", loss = "mean_squared_error", metrics = ["accuracy"])

# Fit the model
model.fit(x = X_train, y = y_train, epochs = 3, batch_size = 128)

```
