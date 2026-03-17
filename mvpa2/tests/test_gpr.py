# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GPR."""

from mvpa2.base import externals
from mvpa2.misc import data_generators
from mvpa2.misc.attrmap import AttributeMap
from mvpa2.kernels.np import GeneralizedLinearKernel

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets
from mvpa2.testing.tools import assert_array_equal, assert_array_almost_equal

skip_if_no_external('scipy') # needed by GPR code
from mvpa2.clfs.gpr import GPR

if __debug__:
    from mvpa2.base import debug


class GPRTests(unittest.TestCase):

    def test_basic(self):
        dataset = data_generators.linear1d_gaussian_noise()
        k = GeneralizedLinearKernel()
        clf = GPR(k)
        clf.train(dataset)
        y = clf.predict(dataset.samples)
        assert_array_equal(y.shape, dataset.targets.shape)

    def test_linear(self):
        pass




