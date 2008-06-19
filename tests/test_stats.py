#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA stats helpers"""

from mvpa.base import externals
from mvpa.clfs.stats import MCNullDist
from mvpa.measures.anova import OneWayAnova
from tests_warehouse import *
# TODO -- use datasets
from mvpa.misc.data_generators import normalFeatureDataset

class StatsTests(unittest.TestCase):

    def testChiSquare(self):
        if not externals.exists('scipy'):
            return

        from mvpa.misc.stats import chisquare

        # test equal distribution
        tbl = N.array([[5,5],[5,5]])
        chi, p = chisquare(tbl)
        self.failUnless( chi == 0.0 )
        self.failUnless( p == 1.0 )

        # test non-equal distribution
        tbl = N.array([[4,0],[0,4]])
        chi, p = chisquare(tbl)
        self.failUnless(chi == 8.0)
        self.failUnless(p < 0.05)


    def testNullDistProb(self):
        ds = normalFeatureDataset(perlabel=20, nlabels=2, nfeatures=2,
                                  means=[[0],[0]], snr=1, nchunks=1)

        null = MCNullDist(permutations=10, tail='right')

        null.fit(OneWayAnova(), ds)

        # check reasonable output (F-score always positive and close to zero
        # for random data
        prob = null.cdf([3,0])
        self.failUnless((prob == [0, 1]).all())
        # has to have matching shape
        self.failUnlessRaises(ValueError, null.cdf, [5,3,4])


    def testDatasetMeasureProb(self):
        ds = normalFeatureDataset(perlabel=20, nlabels=2, nfeatures=4,
                                  nonbogus_features=[1,3], snr=4, nchunks=1)

        # to estimate null distribution
        m = OneWayAnova(null_dist=MCNullDist(permutations=10, tail='right'))

        score = m(ds)

        # plausability check
        self.failUnless(score[0] + score[2] < score[1] + score[3])

        # feature 1 and 3 should have a very unlikely value
        self.failUnless(m.null_prob[1] < 0.05)
        self.failUnless(m.null_prob[3] < 0.05)

        # the others should be a lot larger
        self.failUnless(m.null_prob[0] + m.null_prob[2] \
                        > m.null_prob[1] + m.null_prob[3])



def suite():
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    import runner

