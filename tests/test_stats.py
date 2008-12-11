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
from mvpa.clfs.stats import MCNullDist, FixedNullDist, NullDist
from mvpa.measures.anova import OneWayAnova
from tests_warehouse import *
from mvpa import cfg

# Prepare few distributions to test
#kwargs = {'permutations':10, 'tail':'any'}
nulldist_sweep = [ MCNullDist(permutations=10, tail='any'),
                   MCNullDist(permutations=10, tail='right')]
if externals.exists('scipy'):
    from mvpa.support.stats import scipy
    nulldist_sweep += [ MCNullDist(scipy.stats.norm, permutations=10, tail='any'),
                        MCNullDist(scipy.stats.norm, permutations=10, tail='right'),
                        MCNullDist(scipy.stats.expon, permutations=10, tail='right'),
                        FixedNullDist(scipy.stats.norm(0, 0.1), tail='any'),
                        FixedNullDist(scipy.stats.norm(0, 0.1), tail='right'),
                        scipy.stats.norm(0, 0.1)
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


    @sweepargs(null=nulldist_sweep[1:])
    def testNullDistProb(self, null):
        if not isinstance(null, NullDist):
            return
        ds = datasets['uni2small']

        null.fit(OneWayAnova(), ds)

        # check reasonable output.
        # p-values for non-bogus features should significantly different,
        # while bogus (0) not
        prob = null.p([3,0,0,0,0,N.nan])
        self.failUnless(N.abs(prob[0]) < 0.01)
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless((N.abs(prob[1:]) > 0.05).all(),
                            msg="Bogus features should have insignificant p."
                            " Got %s" % (N.abs(prob[1:]),))

        # has to have matching shape
        if not isinstance(null, FixedNullDist):
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
        if not externals.exists('scipy'):
            # due to null_t requirement
            return

        ds = datasets['uni2medium']

        m = OneWayAnova(null_dist=nd, enable_states=['null_t'])
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
        self.failUnless(N.mean(N.abs(null_prob_bogus)) > N.mean(N.abs(null_prob_nonbogus)))

        if isinstance(nd, MCNullDist):
            # MCs are not stable with just 10 samples... so lets skip them
            return

        self.failUnless((N.abs(m.null_t[ds.nonbogus_features]) >= 5).all(),
            msg="Nonbogus features should have high t-score. Got %s"
                % (m.null_t[ds.nonbogus_features]))

        self.failUnless((N.abs(m.null_t[ds.bogus_features]) < 4).all(),
            msg="Bogus features should have low t-score. Got (t,p,sens):%s"
                % (zip(m.null_t, m.null_prob, score)))


    def testNegativeT(self):
        """Aims to provide very basic testing of the sign in p and t scores
        """
        from mvpa.measures.base import FeaturewiseDatasetMeasure

        class BogusMeasure(FeaturewiseDatasetMeasure):
            """Just put high positive into first 2 features, and high negative into 2nd two
            """
            def _call(self, dataset):
                res = N.random.normal(size=(dataset.nfeatures,))
                res[0] = res[1] = 100
                res[2] = res[3] = -100
                return res
        if not externals.exists('scipy'):
            return
        nd = FixedNullDist(scipy.stats.norm(0, 0.1), tail='any')
        m = BogusMeasure(null_dist=nd, enable_states=['null_t'])
        ds = datasets['uni2small']
        score = m(ds)
        t,p = m.null_t, m.null_prob
        self.failUnless((p>=0).all())
        self.failUnless((t[:2] > 0).all())
        self.failUnless((t[2:4] < 0).all())


    def testMatchDistribution(self):
        """Some really basic testing for matchDistribution
        """
        if not externals.exists('scipy'):
            return
        from mvpa.clfs.stats import matchDistribution, rv_semifrozen
        import scipy.stats

        data = datasets['uni2small'].samples[:,1]

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

        data_mean = N.mean(data)
        for loc in [None, data_mean]:
            for test in ['p-roc', 'kstest']:
                # some really basic testing
                matched = matchDistribution(data=data,
                                            distributions = ['scipy',
                                                             ('norm',
                                                              {'name': 'norm_fixed',
                                                               'loc': 0.2,
                                                               'scale': 0.3})],
                                            test=test, loc=loc, p=0.05)
                # at least norm should be in there
                names = [m[2] for m in matched]
                if test == 'p-roc':
                    if cfg.getboolean('tests', 'labile', default='yes'):
                        # we can guarantee that only for norm_fixed
                        self.failUnless('norm' in names)
                        self.failUnless('norm_fixed' in names)
                        inorm = names.index('norm_fixed')
                        # and it should be at least in the first 30 best matching ;-)
                        self.failUnless(inorm <= 30)


    def testRDistStability(self):
        if not externals.exists('scipy'):
            return

        from mvpa.support.stats import scipy
        try:
            # actually I haven't managed to cause this error
            value = scipy.stats.rdist(1.32, 0, 1).pdf(-1.0+N.finfo(float).eps)
        except:
            self.fail('Failed to compute rdist.pdf due to numeric'
                      ' loss of precision')

        try:
            # this one should fail with recent scipy with error
            # ZeroDivisionError: 0.0 cannot be raised to a negative power

            # XXX: There is 1 more bug in etch's scipy.stats or numpy (vectorize), so I
            #      have to put 2 elements in the queried x's, otherwise it
            #      would puke. But for now that fix is not here
            #
            # value = scipy.stats.rdist(1.32, 0, 1).cdf([-1.0+N.finfo(float).eps, 0])
            #
            # to cause it now just run this unittest only with
            #  nosetests -s test_stats:StatsTests.testRDistStability

            # NB: very cool way to store the trace of the execution
            #import pydb
            #pydb.debugger(dbg_cmds=['bt', 'l', 's']*300 + ['c'])
            value = scipy.stats.rdist(1.32, 0, 1).cdf(-1.0+N.finfo(float).eps)
        except IndexError, e:
            self.fail('Failed due to bug which leads to InvalidIndex if only'
                      ' scalar is provided to cdf')
        except Exception, e:
            self.fail('Failed to compute rdist.cdf due to numeric'
                      ' loss of precision. Exception was %s' % e)

        v = scipy.stats.rdist(10000,0,1).cdf([-0.1])
        self.failUnless(v>=0, v<=1)


def suite():
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    import runner

