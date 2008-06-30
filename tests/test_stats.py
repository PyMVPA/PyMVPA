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
        ds = datasets['uni2small']

        null = MCNullDist(permutations=10, tail='right')

        null.fit(OneWayAnova(), ds)

        # check reasonable output (F-score always positive and close to zero
        # for random data
        prob = null.cdf([3,0,0,0,0,0])
        self.failUnless((prob == [0, 1, 1, 1, 1, 1]).all())
        # has to have matching shape
        self.failUnlessRaises(ValueError, null.cdf, [5,3,4])


    def testDatasetMeasureProb(self):
        ds = datasets['uni2medium']

        # to estimate null distribution
        m = OneWayAnova(null_dist=MCNullDist(permutations=10, tail='right'))

        score = m(ds)

        score_nonbogus = N.mean(score[ds.nonbogus_features])
        score_bogus = N.mean(score[ds.bogus_features])
        # plausability check
        self.failUnless(score_bogus < score_nonbogus)

        null_prob_nonbogus = m.null_prob[ds.nonbogus_features]
        null_prob_bogus = m.null_prob[ds.bogus_features]

        self.failUnless((null_prob_nonbogus < 0.05).all(),
            msg="Nonbogus features should have a very unlikely value. Got %s"
                % null_prob_nonbogus)

        # the others should be a lot larger
        self.failUnless(N.mean(null_prob_bogus) > N.mean(null_prob_nonbogus))



def suite():
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    import runner

