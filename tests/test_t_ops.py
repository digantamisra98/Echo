import unittest

import torch
from parameterized import parameterized

from echoAI.Activation.t_ops import *
from deeptest.functional import *

TEST_SILU = [
    Swish(swish=False),
    'x * sigmoid(x)',
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    (1, 1, 2, 5),
]

TEST_MISH = [
    Mish(),
    'x * tanh(log(1 + e^x))',
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    (1, 1, 2, 5),
]



class TestActivations(unittest.TestCase):

    @parameterized.expand([TEST_SILU, TEST_MISH])
    def test_activations_value_shape(self, fn, fn_expr, xs, shape):
        tester.test(fn, fn_expr, xs, shape)


if __name__ == "__main__":
    tester = WolframTester('QYU645-4EGHX3JVLE', 'torch')
    unittest.main()
