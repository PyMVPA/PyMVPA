#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA null hypothesis testing."""

import unittest

import numpy as N

from mvpa.clfs.svm import LinearNuSVMC
from mvpa.clfs.knn import kNN
from mvpa.algorithms.nullhyptest import NullHypothesisTest
from mvpa.clfs.transerror import TransferError

from tests_warehouse import pureMultivariateSignal

class NullHypothesisTests(unittest.TestCase):

    def testNullHypothesisTest(self):
        wdata = pureMultivariateSignal(10)
        tdata = pureMultivariateSignal(10)

        orig_labels = wdata.labels.copy()

        # linear clf on non-linear problem
        terr = TransferError(LinearNuSVMC())
        null = NullHypothesisTest(terr, permutations=100)

        lin_p = null(wdata, tdata)

        # null distribution must have mean 0.5
        self.failUnless(N.abs(N.mean(null.null_errors-0.5)) < 0.15)

        # must not alter orig labels
        self.failUnless((wdata.labels == orig_labels).all())

        # non-linear clf on non-linear problem
        terr = TransferError(kNN(5))
        null = NullHypothesisTest(terr, permutations=100)

        nlin_p = null(wdata, tdata)

        # null distribution must have mean 0.5
        self.failUnless(N.abs(null.null_errors-0.5).mean() < 0.15)

        # non-linear must perform better than linear
        # Michael: disabled as not deterministic
        #self.failUnless(nlin_p < lin_p)


def suite():
    return unittest.makeSuite(NullHypothesisTests)


if __name__ == '__main__':
    import test_runner

