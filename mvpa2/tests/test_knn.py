# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA kNN classifier"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.datasets import pure_multivariate_signal

from mvpa2.clfs.knn import kNN
from mvpa2.clfs.distance import one_minus_correlation

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
            mv_perf.append( np.mean(p_mv==test.targets) )

            clf.train(train[:, 0])
            p_uv = clf.predict(test[:, 0].samples)
            uv_perf.append( np.mean(p_uv==test.targets) )

        mean_mv_perf = np.mean(mv_perf)
        mean_uv_perf = np.mean(uv_perf)

        self.assertTrue( mean_mv_perf > 0.9 )
        self.assertTrue( mean_uv_perf < mean_mv_perf )


    def test_knn_state(self):
        train = pure_multivariate_signal( 40, 3 )
        test = pure_multivariate_signal( 20, 3 )

        clf = kNN(k=10)
        clf.train(train)

        clf.ca.enable(['estimates', 'predictions', 'distances'])

        p = clf.predict(test.samples)

        self.assertTrue(p == clf.ca.predictions)
        self.assertTrue(len(clf.ca.estimates) == 80)
        self.assertTrue(set(clf.ca.estimates[0].keys()) == set(test.targets))
        self.assertTrue(clf.ca.distances.shape == (80,160))

        self.assertTrue(not clf.ca.distances.fa is train.sa)
        # Those are deep-copied now by default so they should not be the same
        self.assertTrue(not (clf.ca.distances.fa['chunks'] is train.sa['chunks']))
        self.assertTrue(not (clf.ca.distances.fa.chunks is train.sa.chunks))

def suite():  # pragma: no cover
    return unittest.makeSuite(KNNTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

