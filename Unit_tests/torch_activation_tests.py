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
from echoAI.Activation.Torch.weightedTanh import WeightedTanh
from echoAI.Activation.Torch.mish import Mish
from echoAI.Activation.Torch.silu import Silu
from echoAI.Activation.Torch.aria2 import Aria2
from echoAI.Activation.Torch.eswish import Eswish
from echoAI.Activation.Torch.swish import Swish
from echoAI.Activation.Torch.beta_mish import BetaMish
from echoAI.Activation.Torch.elish import Elish
from echoAI.Activation.Torch.hard_elish import HardElish
from echoAI.Activation.Torch.mila import Mila
from echoAI.Activation.Torch.sine_relu import SineReLU
from echoAI.Activation.Torch.fts import FTS
from echoAI.Activation.Torch.sqnl import SQNL
from echoAI.Activation.Torch.isru import ISRU
from echoAI.Activation.Torch.isrlu import ISRLU
from echoAI.Activation.Torch.bent_id import BentID
from echoAI.Activation.Torch.soft_clipping import SoftClipping
import echoAI.Activation.Torch.functional as Func

# class containing unit tests for PyTorch activation functions
class TestTorchActivations(TestCase):
    '''
    Class containing unit tests for PyTorch activation functions
    '''

    def test_weightedTanh_1(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.WeightedTanh`.
        '''
        weighted_tanh = WeightedTanh(weight = 1)
        input = torch.tensor((2.0,2.0))
        # checking that weighted tahn with weight equal to 1 is equal to pytorch weighted tanh
        self.assertEqual(torch.all(torch.eq(weighted_tanh(input), torch.tanh(input))) , True)

    def test_weightedTanh_2(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.WeightedTanh`.
        '''
        weighted_tanh = WeightedTanh(weight = 5.0)
        input = torch.tensor((0.0,0.0))
        output = torch.tensor((0.0,0.0))
        # checking that weighted tahn of 0 is 0
        self.assertEqual(torch.all(torch.eq(weighted_tanh(input), output)) , True)

    def test_weightedTanh_3(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.WeightedTanh`.
        '''
        weighted_tanh = WeightedTanh(weight = 2.0)
        input = torch.tensor((1.1,1.1))
        output = torch.tensor((0.975743,0.975743))
        # checking that weighted tahn of 2.2 is 0.975743
        self.assertEqual((weighted_tanh(input)).allclose(output) , True)

    def test_weightedTanh_4(self):
        '''
        Unit test for weighted tanh activation function.
        See :mod:`Echo.Activation.Torch.WeightedTanh`.
        '''
        weighted_tanh = WeightedTanh(weight = 2.0, inplace = True)
        input = torch.tensor((1.1,1.1))
        output = torch.tensor((0.975743,0.975743))
        # check the inplace implementation
        # checking that weighted tahn of 2.2 is 0.975743
        weighted_tanh(input)
        self.assertEqual((input).allclose(output) , True)

    def test_silu_1(self):
        '''
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.Silu`.
        '''
        fsilu = Silu()
        input = torch.tensor((0.0,0.0))
        output = torch.tensor((0.0,0.0))
        # checking that silu(0) == 0
        self.assertEqual((fsilu(input)).allclose(output) , True)

    def test_silu_2(self):
        '''
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.Silu`.
        '''
        fsilu = Silu()
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.731058,0.731058))

        # checking that silu(1.0) == 0.7310
        self.assertEqual((fsilu(input)).allclose(output) , True)

    def test_silu_3(self):
        '''
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.Silu`.
        '''
        # checking an inplace implementation
        fsilu = Silu(inplace = True)
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.731058,0.731058))

        # performing the inplace operation
        fsilu(input)

        # checking that value of an input is SiLU(input) == 0.7310 now
        self.assertEqual((input).allclose(output) , True)

    def test_mish_1(self):
        '''
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.Mish`.
        '''
        # checking that mish(0) == 0
        fmish = Mish()
        input = torch.tensor((0.0,0.0))
        output = torch.tensor((0.0,0.0))

        self.assertEqual((fmish(input)).allclose(output) , True)

    def test_mish_2(self):
        '''
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.Mish`.
        '''
        # checking that mish(1) == 0.865098
        fmish = Mish()
        input = torch.tensor((1.0,1.0))
        output = torch.tensor((0.865098,0.865098))

        self.assertEqual((fmish(input)).allclose(output) , True)

    def test_aria2_1(self):
        '''
        Unit test for Aria2 activation function.
        See :mod:`Echo.Activation.Torch.Aria2`.
        '''
        # checking that aria2(0, 0) = (0.5, 0.5)
        input = torch.tensor((.0,.0))
        aria = Aria2(beta=1., alpha=1.)
        output = aria(input)

        self.assertEqual(output.allclose(torch.tensor((.5,.5))), True)

    def test_aria2_2(self):
        '''
        Unit test for Aria2 activation function.
        See :mod:`Echo.Activation.Torch.Aria2`.
        '''
        # checking that aria2(1., 1.) = (0.73105, 0.73105)
        input = torch.tensor((1.0,1.0))
        aria = Aria2(beta=1., alpha=1.)
        output = aria(input)

        self.assertEqual(output.allclose(torch.tensor((0.7310585786,0.7310585786))), True)

    def test_beta_mish_1(self):
        '''
        Unit test for Beta Mish activation function.
        See :mod:`Echo.Activation.Torch.BetaMish`.
        '''
        # checking that beta_mish(0., 0.) = (0. , 0.)
        input = torch.tensor((0.0,0.0))
        beta_mish_activation = BetaMish(beta=1.)
        output = beta_mish_activation(input)

        self.assertEqual(output.allclose(torch.tensor((0.0,0.0))), True)

    def test_beta_mish_2(self):
        '''
        Unit test for Beta Mish activation function.
        See :mod:`Echo.Activation.Torch.BetaMish`.
        '''
        # checking that beta_mish(1., 1.) = (0.865098, 0.865098) beta = 1.0
        input = torch.tensor((1.0,1.0))
        beta_mish_activation = BetaMish(beta=1.)
        output = beta_mish_activation(input)

        self.assertEqual(output.allclose(torch.tensor((0.865098,0.865098))), True)

# define entry point
if __name__ == '__main__':
    unittest.main()
