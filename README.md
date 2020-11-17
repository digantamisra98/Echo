<p align="left">
  <img width="270" src="https://github.com/digantamisra98/Echo/raw/master/Observations/logo_transparent.png">
</p>

[![Donate](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3b9607d06bc0420ebe1ce4443e34e1ba)](https://www.codacy.com/manual/digantamisra98/Echo?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=digantamisra98/Echo&amp;utm_campaign=Badge_Grade)
[![HitCount](http://hits.dwyl.io/digantamisra98/Echo.svg)](http://hits.dwyl.io/digantamisra98/Echo)
[![Build Status](https://travis-ci.com/digantamisra98/Echo.svg?branch=master)](https://travis-ci.com/digantamisra98/Echo)
[![codecov](https://codecov.io/gh/digantamisra98/Echo/branch/master/graph/badge.svg)](https://codecov.io/gh/digantamisra98/Echo)

# Echo-AI

*Currently under development. Next release to include activations, optimizers and attention layers.*

Python package containing all mathematical backend algorithms used in Machine Learning.
The full documentation for Echo is provided [here](https://echo-ai.readthedocs.io/en/latest/).

## Table of Contents
* [About the Project](#about)
  * [Activation Functions](#activation-functions)
* [Repository Structure](#repository-structure)
* [Setup Instructions](#setup-instructions)

## About
**Echo-AI Package** is created to provide an implementation of the most promising mathematical algorithms, which are missing in the most popular deep learning libraries, such as [PyTorch](https://pytorch.org/), [Keras](https://keras.io/) and
[TensorFlow](https://www.tensorflow.org/).


## Repository Structure
The repository has the following structure:
```python
- echoAI # main package directory
| - Activation # sub-package containing activation functions implementation
| |- Torch  # sub-package containing implementation for PyTorch
| | | - functional.py # script which contains implementation of activation functions
| | | - weightedTanh.py # activation functions wrapper class for PyTorch
| | | - ... # PyTorch activation functions wrappers
| |- TF_Keras  # sub-package containing implementation for Tensorflow-Keras
| | | - custom_activation.py # script which contains implementation of activation functions
| - __init__.py

- Observations # Folder containing other assets

- docs # Sphinx documentation folder

- LICENSE # license file
- README.md
- setup.py # package setup file
- Scripts #folder, which contains the Black and Flake8 automated test scripts
- Smoke_tests # folder, which contains scripts with demonstration of activation functions usage
- Unit_tests # folder, which contains unit test scripts
```

## Setup Instructions
To install __echoAI__ package from PyPI run the following command:

  ```$ pip install echoAI ```

