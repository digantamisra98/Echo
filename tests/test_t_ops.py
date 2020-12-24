import os
import unittest

import torch
from deeptest.functional import *
from fastcore.script import *
from parameterized import parameterized

from echoAI.Activation.t_ops import *

TEST_SILU = [
    Swish(swish=False),
    "x * sigmoid(x)",
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    (1, 1, 2, 5),
]

TEST_MISH = [
    Mish(),
    "x * tanh(log(1 + e^x))",
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    (1, 1, 2, 5),
]


class TestActivations(unittest.TestCase):

    tester = WolframTester(os.environ.get("WOLFRAM_API_KEY"), "torch")

    @parameterized.expand([TEST_SILU, TEST_MISH])
    def test_activations_value_shape(self, fn, fn_expr, xs, shape):
        self.tester.test(fn, fn_expr, xs, shape)


@call_parse
def main():
    unittest.main()
