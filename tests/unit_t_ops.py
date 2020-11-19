"""
Script containing unit tests for PyTorch activation functions.
"""

import sys

# import unit tests
import unittest
from unittest import TestCase

# import pytorch
import torch

# import custom activations from Echo
from echoAI.Activation.Torch.mish import Mish
from echoAI.Activation.Torch.silu import Silu
from echoAI.Activation.Torch.aria2 import Aria2

sys.path.insert(0, "../")


# class containing unit tests for PyTorch activation functions
class TestTorchActivations(TestCase):
    """
    Class containing unit tests for PyTorch activation functions
    """

    def test_silu_1(self):
        """
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.Silu`.
        """
        fsilu = Silu()
        input = torch.tensor((0.0, 0.0))
        output = torch.tensor((0.0, 0.0))
        # checking that silu(0) == 0
        self.assertEqual((fsilu(input)).allclose(output), True)

    def test_silu_2(self):
        """
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.Silu`.
        """
        fsilu = Silu()
        input = torch.tensor((1.0, 1.0))
        output = torch.tensor((0.731058, 0.731058))

        # checking that silu(1.0) == 0.7310
        self.assertEqual((fsilu(input)).allclose(output), True)

    def test_silu_3(self):
        """
        Unit test for SiLU activation function.
        See :mod:`Echo.Activation.Torch.Silu`.
        """
        # checking an inplace implementation
        fsilu = Silu(inplace=True)
        input = torch.tensor((1.0, 1.0))
        output = torch.tensor((0.731058, 0.731058))

        # performing the inplace operation
        fsilu(input)

        # checking that value of an input is SiLU(input) == 0.7310 now
        self.assertEqual((input).allclose(output), True)

    def test_mish_1(self):
        """
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.Mish`.
        """
        # checking that mish(0) == 0
        fmish = Mish()
        input = torch.tensor((0.0, 0.0))
        output = torch.tensor((0.0, 0.0))

        self.assertEqual((fmish(input)).allclose(output), True)

    def test_mish_2(self):
        """
        Unit test for Mish activation function.
        See :mod:`Echo.Activation.Torch.Mish`.
        """
        # checking that mish(1) == 0.865098
        fmish = Mish()
        input = torch.tensor((1.0, 1.0))
        output = torch.tensor((0.865098, 0.865098))

        self.assertEqual((fmish(input)).allclose(output), True)

    def test_aria2_1(self):
        """
        Unit test for Aria2 activation function.
        See :mod:`Echo.Activation.Torch.Aria2`.
        """
        # checking that aria2(0, 0) = (0.5, 0.5)
        input = torch.tensor((0.0, 0.0))
        aria = Aria2(beta=1.0, alpha=1.0)
        output = aria(input)

        self.assertEqual(output.allclose(torch.tensor((0.5, 0.5))), True)

    def test_aria2_2(self):
        """
        Unit test for Aria2 activation function.
        See :mod:`Echo.Activation.Torch.Aria2`.
        """
        # checking that aria2(1., 1.) = (0.73105, 0.73105)
        input = torch.tensor((1.0, 1.0))
        aria = Aria2(beta=1.0, alpha=1.0)
        output = aria(input)

        self.assertEqual(
            output.allclose(torch.tensor((0.7310585786, 0.7310585786))), True
        )


# define entry point
if __name__ == "__main__":
    unittest.main()
