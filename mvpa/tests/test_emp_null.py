"""
Test the empirical null estimator.

Borrowed from NiPy -- see COPYING distributed with PyMVPA for the
copyright/license information.
"""
import warnings

import numpy as np

from mvpa.base import cfg
from mvpa.testing import *
skip_if_no_external('scipy')
from mvpa.support.nipy import emp_null
#from emp_null import ENN

def setup():
    # Suppress warnings during tests to reduce noise
    warnings.simplefilter("ignore")

def teardown():
    # Clear list of warning filters
    warnings.resetwarnings()

def test_efdr():
    # generate the data
    n = 100000
    x = np.random.randn(n)
    x[:3000] += 3
    #
    # make the tests
    efdr = emp_null.ENN(x)
    np.testing.assert_array_less(efdr.fdr(3.0), 0.15)
    if cfg.getboolean('tests', 'labile', default='yes'):
        np.testing.assert_array_less(-efdr.threshold(alpha=0.05), -3)
    np.testing.assert_array_less(-efdr.uncorrected_threshold(alpha=0.001), -3)

