# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA kernels"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets

from mvpa2.base.externals import exists
from mvpa2.datasets import Dataset
from mvpa2.clfs.distance import squared_euclidean_distance, \
     pnorm_w, pnorm_w_python

import mvpa2.kernels.np as npK
from mvpa2.kernels.base import PrecomputedKernel, CachedKernel
try:
    import mvpa2.kernels.sg as sgK
    _has_sg = exists('shogun')
except RuntimeError:
    _has_sg = False


class KernelTests(unittest.TestCase):
    """Test bloody kernels
    """

    # mvpa2.kernel stuff

    def kernel_equiv(self, k1, k2, accuracy=None, relative_precision=0.6):
        """Test how accurately two kernels agree

        Parameters
        ----------
        k1 : kernel
        k2 : kernel
        accuracy : None or float
          To what accuracy to operate.  If None, length of mantissa
          (precision) is taken into account together with
          relative_precision to provide the `accuracy`.
        relative_precision : float, optional
          What proportion of leading digits in mantissa should match
          between k1 and k2 (effective only if `precision` is None).
        """
        k1m = k1.as_np()._k.copy() #; k1.as_np()._k.setflags(write=0)
        k2m = k2.as_np()._k.copy() #; k2.as_np()._k.setflags(write=0)

        # We should operate on mantissas (given exponents are the same) since
        # pure difference makes no sense to compare and we care about
        # digits in mantissa but there is no convenient way to compare
        # by mantissa:
        #      unfortunately there is no assert_array_approx_equal so
        #      we could specify number of significant digits to use.
        #      assert_array_almost_equal relies on number of decimals AFTER
        #      comma, so both
        #       assert_array_almost_equal([11111.001], [11111.002], decimal=4)
        #       and
        #       assert_array_almost_equal([0.001], [0.002], decimal=4)
        #      would fail, whenever
        #       assert_approx_equal(11111.001, 11111.002, significant=3)
        #      would be ok

        # assert_array_almost_equal(k1m, k2m, decimal=6)
        # assert_approx_equal(k1m, k2m, significant=12)

        if accuracy is None:
            # What precision should be operate at given relative_precision
            # and current dtype

            # first check if dtypes are the same
            ok_(k1m.dtype is k2m.dtype)

            k12mean = 0.5 * (np.abs(k1m) + np.abs(k2m))
            scales = np.ones(k12mean.shape)

            # don't bother dealing with values which would be within
            # resolution -- ** operation would lead to NaNs or 0s
            k12mean_nz = k12mean >= np.finfo(k1m.dtype).resolution * 1e+1
            scales[k12mean_nz] = 10 ** np.floor(np.log10(k12mean[k12mean_nz]))
            for a in (k1m, k2m):
                # lets normalize by exponent first
                anz = a != 0
                # "remove" exponent
                a[anz] /= scales[anz]

            accuracy = 10 ** -(np.finfo(k1m.dtype).precision * relative_precision)

        diff = np.abs(k1m - k2m)
        dmax = diff.max()               # and maximal difference
        dmax_index = np.unravel_index(np.argmax(diff), diff.shape)

        self.assertTrue(dmax <= accuracy,
                        '\n%s\nand\n%s\ndiffer by %s at %s:\n  %.15e\n  %.15e'
                        % (k1, k2, dmax, dmax_index,
                           k1.as_np()._k.__getitem__(dmax_index),
                           k2.as_np()._k.__getitem__(dmax_index)))

        self.assertTrue(np.all(k1m.astype('float32') == \
                              k2m.astype('float32')),
                        '\n%s\nand\n%s\nare unequal as float32' % (k1, k2))


    def test_linear_kernel(self):
        """Simplistic testing of linear kernel"""
        d1 = Dataset(np.asarray([range(5)] * 10, dtype=float))
        lk = npK.LinearKernel()
        lk.compute(d1)
        self.assertTrue(lk._k.shape == (10, 10),
                        "Failure computing LinearKernel (Size mismatch)")
        self.assertTrue((lk._k == 30).all(),
                        "Failure computing LinearKernel")

    @reseed_rng()
    def test_precomputed_kernel(self):
        """Statistic Kernels"""
        d = np.random.randn(50, 50)
        nk = PrecomputedKernel(matrix=d)
        nk.compute()
        self.assertTrue((d == nk._k).all(),
                        'Failure setting and retrieving PrecomputedKernel data')

    @reseed_rng()
    def test_cached_kernel(self):
        nchunks = 5
        n = 50 * nchunks
        d = Dataset(np.random.randn(n, 132))
        d.sa.chunks = np.random.randint(nchunks, size=n)

        # We'll compare against an Rbf just because it has a parameter to change
        rk = npK.RbfKernel(sigma=1.5)

        # Assure two kernels are independent for this test
        ck = CachedKernel(kernel=npK.RbfKernel(sigma=1.5))
        ck.compute(d) # Initial cache of all data

        self.assertTrue(ck._recomputed,
                        'CachedKernel was not initially computed')

        # Try some splitting
        for chunk in [d[d.sa.chunks == i] for i in range(nchunks)]:
            rk.compute(chunk)
            ck.compute(chunk)
            self.kernel_equiv(rk, ck) #, accuracy=1e-12)
            self.failIf(ck._recomputed,
                        "CachedKernel incorrectly recomputed it's kernel")

        # Test what happens when a parameter changes
        ck.params.sigma = 3.5
        ck.compute(d)
        self.assertTrue(ck._recomputed,
                        "CachedKernel doesn't recompute on kernel change")
        rk.params.sigma = 3.5
        rk.compute(d)
        self.assertTrue(np.all(rk._k == ck._k),
                        'Cached and rbf kernels disagree after kernel change')

        # Now test handling new data
        d2 = Dataset(np.random.randn(32, 43))
        ck.compute(d2)
        self.assertTrue(ck._recomputed,
                        "CachedKernel did not automatically recompute new data")
        ck.compute(d)
        self.assertTrue(ck._recomputed,
                        "CachedKernel did not recompute old data which had\n" + \
                        "previously been computed, but had the cache overriden")

    if _has_sg:
        # Unit tests which require shogun kernels
        # Note - there is a loss of precision from double to float32 in SG
        # Not clear if this is just for CustomKernels as there are some
        # remaining innaccuracies in others, but this might be due to other
        # sources of noise.  In all cases float32 should be identical

        @reseed_rng()
        def test_sg_conversions(self):
            nk = PrecomputedKernel(matrix=np.random.randn(50, 50))
            nk.compute()

            skip_if_no_external('shogun',
                                ver_dep='shogun:rev', min_version=4455)
            sk = nk.as_sg()
            sk.compute()
            # CustomKernels interally store as float32 ??
            self.assertTrue((nk._k.astype('float32') == \
                             sk.as_raw_np().astype('float32')).all(),
                            'Failure converting arrays between NP as SG')

        @reseed_rng()
        def test_linear_sg(self):
            d1 = np.random.randn(105, 32)
            d2 = np.random.randn(41, 32)

            nk = npK.LinearKernel()
            sk = sgK.LinearSGKernel()
            nk.compute(d1, d2)
            sk.compute(d1, d2)

            self.kernel_equiv(nk, sk)

        @reseed_rng()
        @labile(5, 1)
        def test_poly_sg(self):
            d1 = np.random.randn(105, 32)
            d2 = np.random.randn(41, 32)
            sk = sgK.PolySGKernel()
            nk = npK.PolyKernel(coef0=1)
            ordervals = [1, 2, 3, 5, 7]
            for p in ordervals:
                sk.params.degree = p
                nk.params.degree = p
                sk.compute(d1, d2)
                nk.compute(d1, d2)

                self.kernel_equiv(nk, sk)

        @reseed_rng()
        def test_rbf_sg(self):
            d1 = np.random.randn(105, 32)
            d2 = np.random.randn(41, 32)
            sk = sgK.RbfSGKernel()
            nk = npK.RbfKernel()
            sigmavals = np.logspace(-2, 5, num=10)
            for s in sigmavals:
                sk.params.sigma = s
                nk.params.sigma = s
                sk.compute(d1, d2)
                nk.compute(d1, d2)

                self.kernel_equiv(nk, sk)

        @reseed_rng()
        def test_custom_sg(self):
            skip_if_no_external('shogun')
            lk = sgK.LinearSGKernel()
            cl = sgK.CustomSGKernel(sgK.sgk.LinearKernel)
            poly = sgK.PolySGKernel()
            poly_params = [('order', 2),
                           ('inhomogenous', True)]
            if not exists('sg ge 0.6.5'):
                poly_params += [ ('use_normalization', False) ]

            custom = sgK.CustomSGKernel(sgK.sgk.PolyKernel,
                                        kernel_params=poly_params)

            d = np.random.randn(253, 52)
            lk.compute(d)
            cl.compute(d)
            poly.compute(d)
            custom.compute(d)

            self.assertTrue(np.all(lk.as_np()._k == cl.as_np()._k),
                            'CustomSGKernel does not agree with Linear')
            self.assertTrue(np.all(poly.as_np()._k == custom.as_np()._k),
                            'CustomSGKernel does not agree with Poly')

    # Older kernel stuff (ie not mvpa2.kernel) - perhaps refactor?
    def test_euclid_dist(self):
        """Euclidean distance kernel testing"""

        # select some block of data from already generated
        data = datasets['uni4large'].samples[:5, :8]

        ed = squared_euclidean_distance(data)

        # XXX not sure if that is right: 'weight' seems to be given by
        # feature (i.e. column), but distance is between samples (i.e. rows)
        # current behavior is:
        true_size = (5, 5)
        self.assertTrue(ed.shape == true_size)

        # slow version to compute distance matrix
        ed_manual = np.zeros(true_size, 'd')
        for i in range(true_size[0]):
            for j in range(true_size[1]):
                #ed_manual[i,j] = np.sqrt(((data[i,:] - data[j,:] )** 2).sum())
                ed_manual[i, j] = ((data[i, :] - data[j, :]) ** 2).sum()
        ed_manual[ed_manual < 0] = 0

        self.assertTrue(np.diag(ed_manual).sum() < 0.0000000001)
        self.assertTrue(np.diag(ed).sum() < 0.0000000001)

        # let see whether Kernel does the same
        self.assertTrue((ed - ed_manual).sum() < 0.0000001)


    def test_pnorm_w(self):
        data0 = datasets['uni4large'].samples.T
        weight = np.abs(data0[11, :60])

        self.assertRaises(ValueError, pnorm_w_python,
                              data0[:10, :2], p=1.2, heuristic='buga')
        self.assertRaises(ValueError, pnorm_w_python,
                              data0[:10, :2], weight=weight)

        self.assertRaises(ValueError, pnorm_w_python,
                              data0[:10, :2], data0[:10, :3],
                              weight=weight)
        self.assertRaises(ValueError, pnorm_w,
                              data0[:10, :2], data0[:10, :3],
                              weight=weight)

        self.assertRaises(ValueError, pnorm_w,
                              data0[:10, :2], weight=weight)

        # some sanity checks
        for did, (data1, data2, w) in enumerate(
            [ (data0[:2, :60], None, None),
              (data0[:2, :60], data0[3:4, 1:61], None),
              (data0[:2, :60], None, weight),
              (data0[:2, :60], data0[3:4, 1:61], weight),
              ]):
            # test different norms
            for p in [1, 2, 1.2]:
                kwargs = {'data1': data1,
                          'data2': data2,
                          'weight' : w,
                          'p' : p}
                d = pnorm_w(**kwargs)    # default one
                # to assess how far we are
                kwargs0 = kwargs.copy()
                kwargs0['data2'] = np.zeros(data1.shape)
                d0 = pnorm_w(**kwargs0)
                d0norm = np.linalg.norm(d - d0, 'fro')
                # test different implementations
                for iid, d2 in enumerate(
                    [pnorm_w_python(**kwargs),
                     pnorm_w_python(use_sq_euclidean=True, **kwargs),
                     pnorm_w_python(heuristic='auto', **kwargs),
                     pnorm_w_python(use_sq_euclidean=False, **kwargs)]
                    +
                    [pnorm_w_python(heuristic=h,
                                    use_sq_euclidean=False, **kwargs)
                     for h in ('auto', 'samples', 'features')]):
                    dnorm = np.linalg.norm(d2 - d, 'fro')
                    self.assertTrue(dnorm / d0norm < 1e-7,
                        msg="Failed comparison of different implementations on "
                            "data #%d, implementation #%d, p=%s. "
                            "Norm of the difference is %g"
                            % (did, iid, p, dnorm))


def suite():  # pragma: no cover
    return unittest.makeSuite(KernelTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

