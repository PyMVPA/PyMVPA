# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA kNN classifier"""

import numpy as N

from mvpa.testing import *
from mvpa.testing.datasets import pure_multivariate_signal

from mvpa.clfs.knn import kNN
from mvpa.clfs.distance import one_minus_correlation

class KNNTests(unittest.TestCase):

    def test_multivariate(self):

        mv_perf = []
        uv_perf = []

        clf = kNN(k=10)
        for i in xrange(20):
            train = pure_multivariate_signal( 20, 3 )
            test = pure_multivariate_signal( 20, 3 )
            clf.train(train)
            p_mv = clf.predict( test.samples )
            mv_perf.append( N.mean(p_mv==test.targets) )

            clf.train(train[:, 0])
            p_uv = clf.predict(test[:, 0].samples)
            uv_perf.append( N.mean(p_uv==test.targets) )

        mean_mv_perf = N.mean(mv_perf)
        mean_uv_perf = N.mean(uv_perf)

        self.failUnless( mean_mv_perf > 0.9 )
        self.failUnless( mean_uv_perf < mean_mv_perf )


    def test_knn_state(self):
        train = pure_multivariate_signal( 40, 3 )
        test = pure_multivariate_signal( 20, 3 )

        clf = kNN(k=10)
        clf.train(train)

        clf.ca.enable(['estimates', 'predictions', 'distances'])

        p = clf.predict(test.samples)

        self.failUnless(p == clf.ca.predictions)
        self.failUnless(N.array(clf.ca.estimates).shape == (80,2))
        self.failUnless(clf.ca.distances.shape == (80,160))

        self.failUnless(not clf.ca.distances.fa is train.sa)
        # XXX if we revert to deepcopy for that state following test
        # should fail
        self.failUnless(clf.ca.distances.fa['chunks'] is train.sa['chunks'])
        self.failUnless(clf.ca.distances.fa.chunks is train.sa.chunks)

def suite():
    return unittest.makeSuite(KNNTests)


if __name__ == '__main__':
    import runner

