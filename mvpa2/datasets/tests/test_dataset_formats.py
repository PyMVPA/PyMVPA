# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''Tests for the dataset implementation'''

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2.datasets.formats import *

import tempfile
import os

def test_format_lightsvm_basic():
    # Just doing basic testing for the silliest of usages -- just
    # dumping / loading data back without any customization via
    # arguments
    for dsname in ['uni2small', 'uni3small', 'chirp_linear']:
        ds = datasets[dsname]
        f = tempfile.NamedTemporaryFile(delete=False)
        am = to_lightsvm_format(ds, f)
        f.close()
        f_ = open(f.name, 'r')
        ds_ = from_lightsvm_format(f_, am=am)
        f_.close()
        os.unlink(f.name)
        # Lets do checks now
        ok_(ds.targets.dtype == ds_.targets.dtype)
        if ds.targets.dtype.char in ['i', 'S', 'U']:
            assert_array_equal(ds.targets, ds_.targets)
        else:
            assert_array_almost_equal(ds.targets, ds_.targets, decimal=3)
        assert_array_almost_equal(ds.samples, ds_.samples)
