'''
Script containing unit tests for PyTorch activation functions.
'''
import sys
sys.path.insert(0, '../')

# import unit tests
import unittest
from unittest import TestCase

# import pytorch
import torch
from torch import nn

# import custom activations from Echo
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

# class containing unit tests for PyTorch activation functions
class TestTorchActivations(TestCase):
    '''
    Class containing unit tests for PyTorch activation functions
    '''

    def test_weightedTanh_1(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.weightedTanh`.
        '''
        weighted_tanh = weightedTanh(weight = 1)
        input = torch.tensor((2.0,2.0))
        # checking that weighted tahn with weight equal to 1 is equal to pytorch weighted tanh
        self.assertEqual(torch.all(torch.eq(weighted_tanh(input), torch.tanh(input))) , True)

    def test_weightedTanh_2(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.weightedTanh`.
        '''
        weighted_tanh = weightedTanh(weight = 5.0)
        input = torch.tensor((0.0,0.0))
        output = torch.tensor((0.0,0.0))
        # checking that weighted tahn of 0 is 0
        self.assertEqual(torch.all(torch.eq(weighted_tanh(input), output)) , True)

    def test_weightedTanh_3(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.weightedTanh`.
        '''
        weighted_tanh = weightedTanh(weight = 2.0)
        input = torch.tensor((1.1,1.1))
        output = torch.tensor((0.975743,0.975743))
        # checking that weighted tahn of 2.2 is 0.975743
        self.assertEqual((weighted_tanh(input)).allclose(output) , True)

# define entry point
if __name__ == '__main__':
    unittest.main()
