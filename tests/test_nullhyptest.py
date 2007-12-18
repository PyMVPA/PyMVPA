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

from mvpa.datasets.dataset import Dataset
from mvpa.clfs.svm import LinearNuSVMC
from mvpa.clfs.knn import kNN
from mvpa.algorithms.nullhyptest import NullHypothesisTest
from mvpa.clfs.transerror import TransferError


def pureMultivariateSignal(patterns, signal2noise = 1.5):
    """ Create a 2d dataset with a clear multivariate signal, but no
    univariate information.

    %%%%%%%%%
    % O % X %
    %%%%%%%%%
    % X % O %
    %%%%%%%%%
    """

    # start with noise
    data=N.random.normal(size=(4*patterns,2))

    # add signal
    data[:2*patterns,1] += signal2noise
    data[2*patterns:4*patterns,1] -= signal2noise
    data[:patterns,0] -= signal2noise
    data[2*patterns:3*patterns,0] -= signal2noise
    data[patterns:2+patterns,0] += signal2noise
    data[3*patterns:4*patterns,0] += signal2noise

    # two conditions
    labels = [0 for i in xrange(patterns)] \
             + [1 for i in xrange(patterns)] \
             + [1 for i in xrange(patterns)] \
             + [0 for i in xrange(patterns)]
    labels = N.array(labels)

    return Dataset(samples=data, labels=labels)


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
        self.failUnless(N.abs(null['null_errors']-0.5).mean() < 0.15)

        # must not alter orig labels
        self.failUnless((wdata.labels == orig_labels).all())

        # non-linear clf on non-linear problem
        terr = TransferError(kNN(5))
        null = NullHypothesisTest(terr, permutations=100)

        nlin_p = null(wdata, tdata)

        # null distribution must have mean 0.5
        self.failUnless(N.abs(null['null_errors']-0.5).mean() < 0.15)

        # non-linear must perform better than linear
        # Michael: disabled as not deterministic
        #self.failUnless(nlin_p < lin_p)


def suite():
    return unittest.makeSuite(NullHypothesisTests)


if __name__ == '__main__':
    import test_runner

