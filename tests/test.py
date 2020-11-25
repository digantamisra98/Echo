from torch.nn.modules import module
from fastcore.script import *
from tests import test_t_ops
import os
import unittest

backend_choices = ['torch', 'tensorflow', 'megengine', 'jax']
module_choices= ['activation', 'attention']

@call_parse
def main(
            backends: Param("DL Frameworks", str, nargs='+', choices=backend_choices),
            modules: Param("Module to test", str, nargs='+', choices=module_choices),
            api_key : Param("Wolfram | Alpha API Key", str) = "QYU645-4EGHX3JVLE"
        ):

    os.environ['WOLFRAM_API_KEY'] = api_key

    suites = list()

    def load_test(suite):
        suite = unittest.TestLoader().loadTestsFromModule(suite)
        suites.append(suite)

    if 'torch' in backends:
        if 'activation' in modules:
            load_test(test_t_ops)

    for suite in suites:
        unittest.TextTestRunner().run(suite)

    return 0