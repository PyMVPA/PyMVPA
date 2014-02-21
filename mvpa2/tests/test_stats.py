# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA stats helpers"""

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2 import cfg
from mvpa2.base import externals
from mvpa2.clfs.stats import MCNullDist, FixedNullDist, NullDist
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.datasets import Dataset
from mvpa2.measures.anova import OneWayAnova, CompoundOneWayAnova
from mvpa2.misc.fx import double_gamma_hrf, single_gamma_hrf
from mvpa2.measures.corrcoef import pearson_correlation

# Prepare few distributions to test
#kwargs = {'permutations':10, 'tail':'any'}
permutator = AttributePermutator('targets', count=30)
nulldist_sweep = [ MCNullDist(permutator, tail='any'),
                   MCNullDist(permutator, tail='right')]

if externals.exists('scipy'):
    from mvpa2.support.scipy.stats import scipy
    from scipy.stats import f_oneway
    from mvpa2.clfs.stats import rv_semifrozen
    nulldist_sweep += [ MCNullDist(permutator, scipy.stats.norm,
                                   tail='any'),
                        MCNullDist(permutator, scipy.stats.norm,
                                   tail='right'),
                        MCNullDist(permutator,
                                   rv_semifrozen(scipy.stats.norm, loc=0),
                                   tail='right'),
                        MCNullDist(permutator, scipy.stats.expon,
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
        prob = null.p([20, 0, 0, 0, 0, np.nan])
        # XXX this is labile! it also needs checking since the F-scores
        # of the MCNullDists using normal distribution are apparently not
        # distributed that way, hence the test often (if not always) fails.
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.assertTrue(np.abs(prob[0]) < 0.05,
                            msg="Expected small p, got %g" % prob[0])
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.assertTrue((np.abs(prob[1:]) > 0.05).all(),
                            msg="Bogus features should have insignificant p."
                            " Got %s" % (np.abs(prob[1:]),))

        # has to have matching shape
        if not isinstance(null, FixedNullDist):
            # Fixed dist is univariate ATM so it doesn't care
            # about dimensionality and gives 1 output value
            self.assertRaises(ValueError, null.p, [5, 3, 4])


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

        self.assertTrue(a.shape == (1, ds.nfeatures))
        self.assertTrue(ac.shape == (len(ds.UT), ds.nfeatures))

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
            self.assertTrue((ac.samples[np.arange(4),
                                        np.array(ds.a.nonbogus_features)] >= 1
                                        ).all())
        # All features should have slightly but different CompoundAnova
        # values. I really doubt that there will be a case when this
        # test would fail just to being 'labile'
        self.assertTrue(np.max(np.std(ac, axis=1)) > 0,
                        msg='In compound anova, we should get different'
                        ' results for different labels. Got %s' % ac)

    def test_pearson_correlation(self):
        sh = (3, -1)
        x = np.reshape(np.asarray([5, 3, 6, 5, 5, 4]), sh)
        y = np.reshape(np.asarray([3, 4, 5, 6, 3, 2, 6, 5, 4, 6, 6, 3]), sh)

        # compute in the traditional way
        nx = x.shape[1]
        ny = y.shape[1]

        c_np = np.zeros((nx, ny))
        for k in xrange(nx):
            for j in xrange(ny):
                c_np[k, j] = np.corrcoef(x[:, k], y[:, j])[0, 1]

        c = pearson_correlation(x, y)

        assert_array_almost_equal(c, c_np)


def suite():  # pragma: no cover
    """Create the suite"""
    return unittest.makeSuite(StatsTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

