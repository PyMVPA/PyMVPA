#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA kNN classifier"""

import unittest

import numpy as N

from mvpa.clfs.knn import kNN

from tests_warehouse import pureMultivariateSignal


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


    def testKNNState(self):
        train = pureMultivariateSignal( 20, 3 )
        test = pureMultivariateSignal( 20, 3 )

        clf = kNN(k=10)
        clf.train(train)

        clf.states.enable('values')
        clf.states.enable('predictions')

        p = clf.predict(test.samples)

        self.failUnless(p == clf.predictions)
        self.failUnless(N.array(clf.values).shape == (80,2))


def suite():
    return unittest.makeSuite(KNNTests)


if __name__ == '__main__':
    import test_runner

