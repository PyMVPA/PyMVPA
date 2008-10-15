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
from mvpa.clfs.stats import MCNullDist, FixedNullDist
from mvpa.measures.anova import OneWayAnova
from tests_warehouse import *

# Prepare few distributions to test
#kwargs = {'permutations':10, 'tail':'any'}
nulldist_sweep = [ MCNullDist(permutations=10, tail='any'),
                   MCNullDist(permutations=10, tail='right')]
if externals.exists('scipy'):
    import scipy.stats
    nulldist_sweep += [ MCNullDist(scipy.stats.norm, permutations=10, tail='any'),
                        MCNullDist(scipy.stats.norm, permutations=10, tail='right'),
                        MCNullDist(scipy.stats.expon, permutations=10, tail='right'),
                        FixedNullDist(scipy.stats.norm(0, 0.01), tail='any'),
                        FixedNullDist(scipy.stats.norm(0, 0.01), tail='right'),
                        ]

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


    @sweepargs(nd=nulldist_sweep[1:])
    def testNullDistProb(self, nd):
        ds = datasets['uni2small']
        null = nd #MCNonparamDist(permutations=10, tail='right')

        null.fit(OneWayAnova(), ds)

        # check reasonable output.
        # p-values for non-bogus features should significantly different,
        # while bogus (0) not
        prob = null.p([3,0,0,0,0,0])
        self.failUnless(prob[0] < 0.01)
        self.failUnless((prob[1:] > 0.05).all())

        # has to have matching shape
        if not isinstance(nd, FixedNullDist):
            # Fixed dist is univariate ATM so it doesn't care
            # about dimensionality and gives 1 output value
            self.failUnlessRaises(ValueError, null.p, [5, 3, 4])


    def testNullDistProbAny(self):
        if not externals.exists('scipy'):
            return

        # test 'any' mode
        from mvpa.measures.corrcoef import CorrCoef
        ds = datasets['uni2small']

        null = MCNullDist(permutations=10, tail='any')
        null.fit(CorrCoef(), ds)

        # 100 and -100 should both have zero probability on their respective
        # tails
        self.failUnless(null.p([-100, 0, 0, 0, 0, 0])[0] == 0)
        self.failUnless(null.p([100, 0, 0, 0, 0, 0])[0] == 0)

        # same test with just scalar measure/feature
        null.fit(CorrCoef(), ds.selectFeatures([0]))
        self.failUnless(null.p(-100) == 0)
        self.failUnless(null.p(100) == 0)


    @sweepargs(nd=nulldist_sweep)
    def testDatasetMeasureProb(self, nd):
        ds = datasets['uni2medium']

        # to estimate null distribution
        m = OneWayAnova(null_dist=nd)

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


    def testMatchDistribution(self):
        """Some really basic testing for matchDistribution
        """
        if not externals.exists('scipy'):
            return
        from mvpa.clfs.stats import matchDistribution, rv_semifrozen
        import scipy.stats

        data = datasets['uni2large'].samples[:,1]

        # Lets test ad-hoc rv_semifrozen
        floc = rv_semifrozen(scipy.stats.norm, loc=0).fit(data)
        self.failUnless(floc[0] == 0)

        fscale = rv_semifrozen(scipy.stats.norm, scale=1.0).fit(data)
        self.failUnless(fscale[1] == 1)

        flocscale = rv_semifrozen(scipy.stats.norm, loc=0, scale=1.0).fit(data)
        self.failUnless(flocscale[1] == 1 and flocscale[0] == 0)

        full = scipy.stats.norm.fit(data)
        for res in [floc, fscale, flocscale, full]:
            self.failUnless(len(res) == 2)

        for loc in [None, N.mean(data)]:
            for test in ['p-roc', 'kstest']:
                # some really basic testing
                matched = matchDistribution(data=data, test=test, loc=loc, p=0.05)
                # at least norm should be in there
                names = [m[1] for m in matched]
                self.failUnless('norm' in names)
                inorm = names.index('norm')
                # and it should be at least in the first 5 best matching
                self.failUnless(inorm <= 7)


def suite():
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    import runner

