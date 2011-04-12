# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA stats helpers -- those requiring scipy"""

from mvpa.testing import *
skip_if_no_external('scipy')

from mvpa.testing.datasets import datasets
from mvpa.tests.test_stats import *

from scipy import signal
from mvpa.clfs.stats import match_distribution, rv_semifrozen
from mvpa.misc.stats import chisquare
from mvpa.misc.attrmap import AttributeMap
from mvpa.datasets.base import dataset_wizard
from mvpa.generators.permutation import AttributePermutator

class StatsTestsScipy(unittest.TestCase):
    """Unittests for various statistics which use scipy"""

    @sweepargs(exp=('uniform', 'indep_rows', 'indep_cols'))
    def test_chi_square(self, exp):
        """Test chi-square distribution"""
        # test equal distribution
        tbl = np.array([[5, 5], [5, 5]])
        chi, p = chisquare(tbl, exp=exp)
        self.failUnless( chi == 0.0 )
        self.failUnless( p == 1.0 )

        # test perfect "generalization"
        tbl = np.array([[4, 0], [0, 4]])
        chi, p = chisquare(tbl, exp=exp)
        self.failUnless(chi == 8.0)
        self.failUnless(p < 0.05)

    def test_chi_square_disbalanced(self):
        # test perfect "generalization"
        tbl = np.array([[1, 100], [1, 100]])
        chi, p = chisquare(tbl, exp='indep_rows')
        self.failUnless(chi == 0)
        self.failUnless(p == 1)

        chi, p = chisquare(tbl, exp='uniform')
        self.failUnless(chi > 194)
        self.failUnless(p < 1e-10)

        # by default lets do uniform
        chi_, p_ = chisquare(tbl)
        self.failUnless(chi == chi_)
        self.failUnless(p == p_)


    def test_null_dist_prob_any(self):
        """Test 'any' tail statistics estimation"""
        skip_if_no_external('scipy')

        # test 'any' mode
        from mvpa.measures.corrcoef import CorrCoef
        ds = datasets['uni2medium']

        permutator = AttributePermutator('targets', count=20)
        null = MCNullDist(permutator, tail='any')

        assert_raises(ValueError, null.fit, CorrCoef(), ds)
        # cheat and map to numeric for this test
        ds.sa.targets = AttributeMap().to_numeric(ds.targets)
        null.fit(CorrCoef(), ds)

        # 100 and -100 should both have zero probability on their respective
        # tails
        pm100 = null.p([-100] + [0]*(ds.nfeatures-1))
        p100 = null.p([100] + [0]*(ds.nfeatures-1))
        assert_array_almost_equal(pm100, p100)

        # With 20 samples it isn't that easy to get a reliable sampling for
        # non-parametric, so we can allow somewhat low significance
        self.failUnless(pm100[0] <= 0.1)
        self.failUnless(p100[0] <= 0.1)

        self.failUnless(np.all(pm100[1:] > 0.05))
        self.failUnless(np.all(p100[1:] > 0.05))
        # same test with just scalar measure/feature
        null.fit(CorrCoef(), ds[:, 0])
        p_100 = null.p(100)
        self.failUnlessAlmostEqual(null.p(-100), p_100)
        self.failUnlessAlmostEqual(p100[0], p_100)


    @sweepargs(nd=nulldist_sweep)
    def test_dataset_measure_prob(self, nd):
        """Test estimation of measures statistics"""
        skip_if_no_external('scipy')

        ds = datasets['uni2medium']

        m = OneWayAnova(null_dist=nd, enable_ca=['null_t'])
        score = m(ds)

        score_nonbogus = np.mean(score.samples[:, ds.a.nonbogus_features])
        score_bogus = np.mean(score.samples[:, ds.a.bogus_features])
        # plausability check
        self.failUnless(score_bogus < score_nonbogus)

        # [0] because the first axis is len == 0
        null_prob_nonbogus = m.ca.null_prob[0, ds.a.nonbogus_features]
        null_prob_bogus = m.ca.null_prob[0, ds.a.bogus_features]

        self.failUnless((null_prob_nonbogus.samples < 0.05).all(),
            msg="Nonbogus features should have a very unlikely value. Got %s"
                % null_prob_nonbogus)

        # the others should be a lot larger
        self.failUnless(np.mean(np.abs(null_prob_bogus)) >
                        np.mean(np.abs(null_prob_nonbogus)))

        if isinstance(nd, MCNullDist):
            # MCs are not stable with just 10 samples... so lets skip them
            return

        if cfg.getboolean('tests', 'labile', default='yes'):
            # Failed on c94ec26eb593687f25d8c27e5cfdc5917e352a69
            # with MVPA_SEED=833393575
            self.failUnless(
                (np.abs(m.ca.null_t[0, ds.a.nonbogus_features]) >= 5).all(),
                msg="Nonbogus features should have high t-score. Got %s"
                % (m.ca.null_t[0, ds.a.nonbogus_features]))

            bogus_min = min(np.abs(m.ca.null_t.samples[0][ds.a.bogus_features]))
            self.failUnless(bogus_min < 4,
                msg="Some bogus features should have low t-score of %g."
                    "Got (t,p,sens):%s"
                    % (bogus_min,
                        zip(m.ca.null_t[0, ds.a.bogus_features],
                            m.ca.null_prob[0, ds.a.bogus_features],
                            score.samples[0][ds.a.bogus_features])))

    @reseed_rng()
    def test_negative_t(self):
        """Basic testing of the sign in p and t scores
        """
        from mvpa.measures.base import FeaturewiseMeasure

        class BogusMeasure(FeaturewiseMeasure):
            """Just put high positive into first 2 features, and high
            negative into 2nd two
            """
            is_trained = True
            def _call(self, dataset):
                """just a little helper... pylint shut up!"""
                res = np.random.normal(size=(dataset.nfeatures,))
                res[0] = res[1] = 100
                res[2] = res[3] = -100
                return Dataset([res])

        nd = FixedNullDist(scipy.stats.norm(0, 0.1), tail='any')
        m = BogusMeasure(null_dist=nd, enable_ca=['null_t'])
        ds = datasets['uni2small']
        _ = m(ds)
        t, p = m.ca.null_t, m.ca.null_prob
        self.failUnless((p.samples>=0).all())
        self.failUnless((t.samples[0,:2] > 0).all())
        self.failUnless((t.samples[0,2:4] < 0).all())


    def test_match_distribution(self):
        """Some really basic testing for match_distribution
        """
        ds = datasets['uni2medium']      # large to get stable stats
        data = ds.samples[:, ds.a.bogus_features[0]]
        # choose bogus feature, which
        # should have close to normal distribution

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

        data_mean = np.mean(data)
        for loc in [None, data_mean]:
            for test in ['p-roc', 'kstest']:
                # some really basic testing
                matched = match_distribution(
                    data=data,
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
                        # and it should be at least in the first
                        # 30 best matching ;-)
                        self.failUnless(inorm <= 30)

    def test_r_dist_stability(self):
        """Test either rdist distribution performs nicely
        """
        try:
            # actually I haven't managed to cause this error
            scipy.stats.rdist(1.32, 0, 1).pdf(-1.0+np.finfo(float).eps)
        except Exception, e:
            self.fail('Failed to compute rdist.pdf due to numeric'
                      ' loss of precision. Exception was %s' % e)

        try:
            # this one should fail with recent scipy with error
            # ZeroDivisionError: 0.0 cannot be raised to a negative power

            # XXX: There is 1 more bug in etch's scipy.stats or numpy
            #      (vectorize), so I have to put 2 elements in the
            #      queried x's, otherwise it
            #      would puke. But for now that fix is not here
            #
            # value = scipy.stats.rdist(1.32, 0, 1).cdf(
            #      [-1.0+np.finfo(float).eps, 0])
            #
            # to cause it now just run this unittest only with
            #  nosetests -s test_stats:StatsTests.testRDistStability

            # NB: very cool way to store the trace of the execution
            #import pydb
            #pydb.debugger(dbg_cmds=['bt', 'l', 's']*300 + ['c'])
            scipy.stats.rdist(1.32, 0, 1).cdf(-1.0+np.finfo(float).eps)
        except IndexError, e:
            self.fail('Failed due to bug which leads to InvalidIndex if only'
                      ' scalar is provided to cdf')
        except Exception, e:
            self.fail('Failed to compute rdist.cdf due to numeric'
                      ' loss of precision. Exception was %s' % e)

        v = scipy.stats.rdist(10000, 0, 1).cdf([-0.1])
        self.failUnless(v>=0, v<=1)


    def test_anova_compliance(self):
        ds = datasets['uni2large']

        fwm = OneWayAnova()
        f = fwm(ds)
        f_sp = f_oneway(ds[ds.targets == 'L1'].samples,
                        ds[ds.targets == 'L0'].samples)

        # SciPy needs to compute the same F-scores
        assert_array_almost_equal(f, f_sp[0:1])


    @reseed_rng()
    def test_glm(self):
        """Test GLM
        """
        # play fmri
        # full-blown HRF with initial dip and undershoot ;-)
        hrf_x = np.linspace(0, 25, 250)
        hrf = double_gamma_hrf(hrf_x) - single_gamma_hrf(hrf_x, 0.8, 1, 0.05)

        # come up with an experimental design
        samples = 1800
        fast_er_onsets = np.array([10, 200, 250, 500, 600, 900, 920, 1400])
        fast_er = np.zeros(samples)
        fast_er[fast_er_onsets] = 1

        # high resolution model of the convolved regressor
        model_hr = np.convolve(fast_er, hrf)[:samples]

        # downsample the regressor to fMRI resolution
        tr = 2.0
        model_lr = signal.resample(model_hr,
                                   int(samples / tr / 10),
                                   window='ham')

        # generate artifical fMRI data: two voxels one is noise, one has
        # something
        baseline = 800.0
        wsignal = baseline + 2 * model_lr + \
                  np.random.randn(int(samples / tr / 10)) * 0.2
        nsignal = baseline + np.random.randn(int(samples / tr / 10)) * 0.5

        # build design matrix: bold-regressor and constant
        X = np.array([model_lr, np.repeat(1, len(model_lr))]).T

        # two 'voxel' dataset
        data = dataset_wizard(samples=np.array((wsignal, nsignal, nsignal)).T, targets=1)

        # check GLM betas
        glm = GLM(X)
        betas = glm(data)

        # betas for each feature and each regressor
        self.failUnless(betas.shape == (X.shape[1], data.nfeatures))

        self.failUnless(np.absolute(betas.samples[1] - baseline < 10).all(),
            msg="baseline betas should be huge and around 800")

        self.failUnless(betas.samples[0,0] > betas[0,1],
            msg="feature (with signal) beta should be larger than for noise")

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(np.absolute(betas[0,1]) < 0.5)
            self.failUnless(np.absolute(betas[0,0]) > 1.0)


        # check GLM zscores
        glm = GLM(X, voi='zstat')
        zstats = glm(data)

        self.failUnless(zstats.shape == betas.shape)

        self.failUnless((zstats.samples[1] > 1000).all(),
                msg='constant zstats should be huge')

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(np.absolute(betas[0,0]) > betas[0,1],
                msg='with signal should have higher zstats')


    def test_binomdist_ppf(self):
        """Test if binomial distribution works ok

        after possibly a monkey patch
        """
        bdist = scipy.stats.binom(100, 0.5)
        self.failUnless(bdist.ppf(1.0) == 100)
        self.failUnless(bdist.ppf(0.9) <= 60)
        self.failUnless(bdist.ppf(0.5) == 50)
        self.failUnless(bdist.ppf(0) == -1)


def suite():
    """Create the suite"""
    return unittest.makeSuite(StatsTestsScipy)


if __name__ == '__main__':
    import runner

