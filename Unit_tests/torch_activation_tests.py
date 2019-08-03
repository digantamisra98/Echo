'''
Script containing unit tests for PyTorch activation functions.
'''
import numpy as np

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

    def test_weightedTanh_4(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.weightedTanh`.
        '''
        weighted_tanh = weightedTanh(weight = 2.0, inplace = True)
        input = torch.tensor((1.1,1.1))
        output = torch.tensor((0.975743,0.975743))
        # check the inplace implementation
        # checking that weighted tahn of 2.2 is 0.975743
        weighted_tanh(input)
        self.assertEqual((input).allclose(output) , True)

    def test_silu_1(self):
        '''
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.silu`.
        '''
        fsilu = silu()
        input = torch.tensor((0.0,0.0))
        output = torch.tensor((0.0,0.0))
        # checking that silu(0) == 0
        self.assertEqual((fsilu(input)).allclose(output) , True)

    def test_silu_2(self):
        '''
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.silu`.
        '''
        fsilu = silu()
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.731058,0.731058))

        # checking that silu(1.0) == 0.7310
        self.assertEqual((fsilu(input)).allclose(output) , True)

    def test_silu_3(self):
        '''
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.silu`.
        '''
        # checking an inplace implementation
        fsilu = silu(inplace = True)
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.731058,0.731058))

        # performing the inplace operation
        fsilu(input)

        # checking that value of an input is SiLU(input) == 0.7310 now
        self.assertEqual((input).allclose(output) , True)

    def test_mish_1(self):
        '''
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.mish`.
        '''
        # checking that mish(0) == 0
        fmish = mish()
        input = torch.tensor((0.0,0.0))
        output = torch.tensor((0.0,0.0))

        self.assertEqual((fmish(input)).allclose(output) , True)

    def test_mish_2(self):
        '''
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.mish`.
        '''
        # checking that mish(1) == 0.865098
        fmish = mish()
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.865098,0.865098))

        self.assertEqual((fmish(input)).allclose(output) , True)

    def test_mish_3(self):
        '''
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.mish`.
        '''
        # checking the in-place implementation of mish
        # checking that mish(1) == 0.865098
        fmish = mish(inplace = True)
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.865098,0.865098))
        fmish(input)

        self.assertEqual((input).allclose(output) , True)

    def test_aria2_1(self):
        '''
        Unit test for Aria2 activation function.
        See :mod:`Echo.Activation.Torch.aria2`.
        '''
        # checking that aria2(0, 0) = (0.5, 0.5)
        input = torch.tensor((.0,.0))
        aria = aria2(beta=1., alpha=1.)
        output = aria(input)

        self.assertEqual(output.allclose(torch.tensor((.5,.5))), True)

    def test_aria2_2(self):
        '''
        Unit test for Aria2 activation function.
        See :mod:`Echo.Activation.Torch.aria2`.
        '''
        # checking that aria2(1., 1.) = (0.73105, 0.73105)
        input = torch.tensor((1.0,1.0))
        aria = aria2(beta=1., alpha=1.)
        output = aria(input)

        print(output)

        self.assertEqual(output.allclose(torch.tensor((0.7310585786,0.7310585786))), True)

# define entry point
if __name__ == '__main__':
    unittest.main()
