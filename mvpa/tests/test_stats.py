# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA stats helpers"""

from mvpa import cfg
from mvpa.base import externals
from mvpa.clfs.stats import MCNullDist, FixedNullDist, NullDist
from mvpa.datasets import Dataset
from mvpa.measures.glm import GLM
from mvpa.measures.anova import OneWayAnova, CompoundOneWayAnova
from mvpa.misc.fx import double_gamma_hrf, single_gamma_hrf
from tests_warehouse import *
from mvpa.testing.tools import assert_array_almost_equal, assert_array_equal, \
assert_true, assert_equal

# Prepare few distributions to test
#kwargs = {'permutations':10, 'tail':'any'}
nulldist_sweep = [ MCNullDist(permutations=30, tail='any'),
                   MCNullDist(permutations=30, tail='right')]

if externals.exists('scipy'):
    from mvpa.support.stats import scipy
    from scipy.stats import f_oneway
    from mvpa.clfs.stats import rv_semifrozen
    nulldist_sweep += [ MCNullDist(scipy.stats.norm, permutations=30,
                                   tail='any'),
                        MCNullDist(scipy.stats.norm, permutations=30,
                                   tail='right'),
                        MCNullDist(rv_semifrozen(scipy.stats.norm, loc=0),
                                   permutations=30, tail='right'),
                        MCNullDist(scipy.stats.expon, permutations=30,
                                   tail='right'),
                        FixedNullDist(scipy.stats.norm(0, 10.0), tail='any'),
                        FixedNullDist(scipy.stats.norm(0, 10.0), tail='right'),
                        scipy.stats.norm(0, 0.1)
                        ]

class StatsTests(unittest.TestCase):
    """Unittests for various statistics"""


    @sweepargs(null=nulldist_sweep[1:])
    def test_null_dist_prob(self, null):
        """Testing null dist probability"""
        if not isinstance(null, NullDist):
            return
        ds = datasets['uni2small']

        null.fit(OneWayAnova(), ds)

        # check reasonable output.
        # p-values for non-bogus features should significantly different,
        # while bogus (0) not
        prob = null.p([20, 0, 0, 0, 0, N.nan])
        # XXX this is labile! it also needs checking since the F-scores
        # of the MCNullDists using normal distribution are apparently not
        # distributed that way, hence the test often (if not always) fails.
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(N.abs(prob[0]) < 0.05,
                            msg="Expected small p, got %g" % prob[0])
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless((N.abs(prob[1:]) > 0.05).all(),
                            msg="Bogus features should have insignificant p."
                            " Got %s" % (N.abs(prob[1:]),))

        # has to have matching shape
        if not isinstance(null, FixedNullDist):
            # Fixed dist is univariate ATM so it doesn't care
            # about dimensionality and gives 1 output value
            self.failUnlessRaises(ValueError, null.p, [5, 3, 4])


    def test_anova(self):
        """Do some extended testing of OneWayAnova

        in particular -- compound estimation
        """

        m = OneWayAnova()               # default must be not compound ?
        mc = CompoundOneWayAnova()
        ds = datasets['uni2medium']

        # For 2 labels it must be identical for both and equal to
        # simple OneWayAnova
        a, ac = m(ds), mc(ds)

        self.failUnless(a.shape == (1, ds.nfeatures))
        self.failUnless(ac.shape == (len(ds.UT), ds.nfeatures))

        assert_array_equal(ac[0], ac[1])
        assert_array_equal(a, ac[1])

        # check for p-value attrs
        if externals.exists('scipy'):
            assert_true('fprob' in a.fa.keys())
            assert_equal(len(ac.fa), len(ac))

        ds = datasets['uni4large']
        ac = mc(ds)
        if cfg.getboolean('tests', 'labile', default='yes'):
            # All non-bogus features must be high for a corresponding feature
            self.failUnless((ac.samples[N.arange(4),
                                        N.array(ds.a.nonbogus_features)] >= 1
                                        ).all())
        # All features should have slightly but different CompoundAnova
        # values. I really doubt that there will be a case when this
        # test would fail just to being 'labile'
        self.failUnless(N.max(N.std(ac, axis=1))>0,
                        msg='In compound anova, we should get different'
                        ' results for different labels. Got %s' % ac)

def suite():
    """Create the suite"""
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':
    import runner

