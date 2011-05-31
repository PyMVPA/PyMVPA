# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA GPR."""

import unittest

from mvpa.misc import data_generators
from mvpa.clfs.kernel import KernelLinear as GeneralizedLinearKernel
from mvpa.clfs.gpr import GPR

from numpy.testing import assert_array_equal, assert_array_almost_equal

if __debug__:
    from mvpa.base import debug
    

class GPRTests(unittest.TestCase):

    def test_basic(self):
        dataset = data_generators.linear1d_gaussian_noise()
        k = GeneralizedLinearKernel()
        clf = GPR(k)
        clf.train(dataset)
        y = clf.predict(dataset.samples)
        assert_array_equal(y.shape, dataset.labels.shape)

    def test_linear(self):
        pass


def suite():
    return unittest.makeSuite(GPRTests)


if __name__ == '__main__':
    import runner
