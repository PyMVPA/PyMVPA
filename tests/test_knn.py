#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Unit tests for PyMVPA kNN classifier"""

import unittest

import numpy as N

from mvpa.datasets.dataset import Dataset
from mvpa.clf.knn import kNN


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

    return Dataset(data, labels, None)


class KNNTests(unittest.TestCase):

    def testMultivariate(self):

        mv_perf = []
        uv_perf = []

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            k_mv = kNN(k = 10)
            k_mv.train(train)
            p_mv = k_mv.predict( test.samples )
            mv_perf.append( N.mean(p_mv==test.labels) )

            k_uv = kNN(k=10)
            k_uv.train(train.selectFeatures([0]))
            p_uv = k_uv.predict( test.selectFeatures([0]).samples )
            uv_perf.append( N.mean(p_uv==test.labels) )

        mean_mv_perf = N.mean(mv_perf)
        mean_uv_perf = N.mean(uv_perf)

        self.failUnless( mean_mv_perf > 0.9 )
        self.failUnless( mean_uv_perf < mean_mv_perf )


def suite():
    return unittest.makeSuite(KNNTests)


if __name__ == '__main__':
    unittest.main()

