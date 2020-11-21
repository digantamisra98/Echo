import unittest

import torch
from parameterized import parameterized

from echoAI.Activation.t_ops import *


TEST_SWISH = [
    Swish(swish=True),
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    torch.tensor(
        [[[[-4.54e-04, -2.68e-03, -1.48e-02, -7.19e-02, -2.38e-01], [0.00e00, 1.76e00, 3.93e00, 5.99e00, 8.00e00]]]]
    ),
    (1, 1, 2, 5),
]

TEST_MISH = [
    Mish(),
    torch.tensor([[[[-10, -8, -6, -4, -2], [0, 2, 4, 6, 8]]]], dtype=torch.float32),
    torch.tensor(
        [[[[-4.54e-04, -2.68e-03, -1.49e-02, -7.26e-02, -2.53e-01], [0.00e00, 1.94e00, 4.00e00, 6.00e00, 8.00e00]]]]
    ),
    (1, 1, 2, 5),
]


class TestActivations(unittest.TestCase):
    @parameterized.expand([TEST_SWISH, TEST_MISH])
    def test_activations_value_shape(self, cls, inp, out, exp):
        result = cls(inp)
        torch.testing.assert_allclose(result, out, rtol=1e-2, atol=1e-5)
        self.assertTupleEqual(result.shape, exp)


if __name__ == "__main__":

    unittest.main()