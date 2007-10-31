#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for SVM classifier"""

import unittest

import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.clf.svm import SVM


def dumbFeatureSignal():
    data = [[1,0],[1,1],[2,0],[2,1],[3,0],[3,1],[4,0],[4,1],
            [5,0],[5,1],[6,0],[6,1],[7,0],[7,1],[8,0],[8,1],
            [9,0],[9,1],[10,0],[10,1],[11,0],[11,1],[12,0],[12,1]]
    regs = [1 for i in range(8)] \
         + [2 for i in range(8)] \
         + [3 for i in range(8)]

    return Dataset(data, regs, None)


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
    regs = [0 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [1 for i in xrange(patterns)] \
        + [0 for i in xrange(patterns)]
    regs = N.array(regs)

    return Dataset(data, regs, None)


class SVMTests(unittest.TestCase):

    def testMultivariate(self):

        mv_perf = []
        uv_perf = []

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            s_mv = SVM()
            s_mv.train(train)
            p_mv = s_mv.predict(test.samples )
            mv_perf.append(N.mean(p_mv==test.labels))

            s_uv = SVM()
            s_uv.train(train.selectFeatures([0]))
            p_uv = s_uv.predict(test.selectFeatures([0]).samples)
            uv_perf.append(N.mean(p_uv==test.labels))

        mean_mv_perf = N.mean(mv_perf)
        mean_uv_perf = N.mean(uv_perf)

        self.failUnless( mean_mv_perf > 0.9 )
        self.failUnless( mean_uv_perf < mean_mv_perf )


    def testFeatureBenchmark(self):
        pat = dumbFeatureSignal()
        clf = SVM()
        clf.train(pat)
        rank = clf.getFeatureBenchmark()

        # has to be 1d array
        self.failUnless(len(rank.shape) == 1)

        # has to be one value per feature
        self.failUnless(len(rank) == pat.nfeatures)

        # first feature is discriminative, second is not
        self.failUnless(rank[0] > rank[1])


def suite():
    return unittest.makeSuite(SVMTests)


if __name__ == '__main__':
    import test_runner

