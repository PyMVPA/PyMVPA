# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA misc stuff"""

from mvpa2.testing import *

from mvpa2.datasets import Dataset
from mvpa2.misc.fx import dual_gaussian, dual_positive_gaussian, fit2histogram
from mvpa2.misc.data_generators import random_affine_transformation

@reseed_rng()
@sweepargs(f=(dual_gaussian, dual_positive_gaussian))
def test_dual_gaussian(f):
    skip_if_no_external('scipy')
    data = np.random.normal(size=(100, 1))

    histfit = fit2histogram(np.repeat(data[None, :], 2, axis=0),
                            f,
                            (1000, 0.5, 0.1, 1000, 0.8, 0.05),
                            nbins=20)
    H, bin_left, bin_width, fit = histfit
    params = fit[0]
    # both variances must be positive
    ok_(params[2] > 0)
    ok_(params[5] > 0)

    if f is dual_positive_gaussian:
        # both amplitudes must be positive
        ok_(params[0] > 0)
        ok_(params[3] > 0)


def test_random_affine_transformation():
    ds = Dataset.from_wizard(np.random.randn(8,3,2))
    ds_d = random_affine_transformation(ds)
    # compare original to the inverse of the distortion using reported
    # parameters
    assert_array_almost_equal(
        np.dot((ds_d.samples - ds_d.a.random_shift) / ds_d.a.random_scale,
               ds_d.a.random_rotation.T),
        ds.samples)


@reseed_rng()
def test_ttest_1samp_masked():
    skip_if_no_external('scipy')
    import numpy as np
    from mvpa2.misc.stats import ttest_1samp as ttest_1samp_masked

    # old scipy's ttest_1samp need to be conditioned since they
    # return 1's and 0's for when should be NaNs
    if externals.versions['scipy'] < '0.10.1':
        from scipy.stats import ttest_1samp as scipy_ttest_1samp

        def ttest_1samp(*args, **kwargs):
            t, p = scipy_ttest_1samp(*args, **kwargs)
            p_isnan = np.isnan(p)
            if np.any(p_isnan):
                if t.ndim == 0:
                    t = np.nan
                else:
                    t[p_isnan] = np.nan
            return t, p
    else:
        from scipy.stats import ttest_1samp

    if externals.versions['numpy'] < '1.6.2':
        # yoh: there is a bug in old (e.g. 1.4.1) numpy's while operating on
        #      masked arrays -- for some reason refuses to compute var
        #      correctly whenever only 2 elements are available and it is
        #      multi-dimensional:
        # (Pydb) print np.var(a[:, 9:11], axis, ddof=1)
        # [540.0 --]
        # (Pydb) print np.var(a[:, 10:11], axis, ddof=1)
        # [--]
        # (Pydb) print np.var(a[:, 10], axis, ddof=1)
        # 648.0
        # To overcome -- assure masks with without 2 elements in any
        # dimension and allow for NaN t-test results in such anyway
        # degenerate cases
        def random_mask(shape):
            # screw it -- let's generate quite primitive mask with
            return (np.arange(np.prod(shape))%2).astype(bool).reshape(shape)
        ndshape = (5, 6, 1, 7)          # we need larger structure with this XOR mask
    else:
        def random_mask(shape):
            # otherwise all simple:
            return np.random.normal(size=shape) > -0.5
        ndshape = (4, 3, 2, 1)

    _assert_array_equal = assert_array_almost_equal

    # test on some random data to match results of ttest_1samp
    d = np.random.normal(size=(5, 3))
    for null in 0, 0.5:
        # 1D case
        _assert_array_equal(ttest_1samp       (d[0], null),
                            ttest_1samp_masked(d[0], null))

        for axis in 0, 1, None:
            _assert_array_equal(ttest_1samp       (d, null, axis=axis),
                                ttest_1samp_masked(d, null, axis=axis))
    # we do not yet support >2D
    ##assert_raises(AssertionError, ttest_1samp_masked, d[None,...], 0)

    # basic test different alternatives
    d = range(10)
    tl, pl = ttest_1samp_masked(d, 0, alternative='greater')

    tr, pr = ttest_1samp_masked(d, 0, alternative='less')
    tb, pb = ttest_1samp_masked(d, 0, alternative='two-sided')

    assert_equal(tl, tr)
    assert_equal(tl, tb)
    assert_equal(pl + pr, 1.0)
    assert_equal(pb, pl*2)
    assert(pl < 0.05)               # clearly we should be able to reject

    # finally let's get to masking
    # 1D
    d = np.arange(10)
    _assert_array_equal(ttest_1samp       (d[3:], 0),
                        ttest_1samp_masked(d,     0,
                                           mask=[False]*3 + [True]*7))

    # random mask
    m = random_mask(d.shape)
    _assert_array_equal(ttest_1samp       (d[m], 0),
                        ttest_1samp_masked(d,    0, mask=m))

    # 2D masking
    d = np.arange(30).reshape((5,-1))
    m = random_mask(d.shape)

    # axis=1
    ts, ps = ttest_1samp_masked(d, 0, mask=m, axis=1)
    for d_, m_, t_, p_ in zip(d, m, ts, ps):
        _assert_array_equal(ttest_1samp (d_[m_], 0), (t_, p_))

    # axis=0
    ts, ps = ttest_1samp_masked(d, 0, mask=m, axis=0)
    for d_, m_, t_, p_ in zip(d.T, m.T, ts, ps):
        _assert_array_equal(ttest_1samp (d_[m_], 0), (t_, p_))

    #5D masking
    d = np.random.normal(size=ndshape)
    m = random_mask(d.shape)

    for axis in range(d.ndim):
        for t0 in (0, 1.0):             # test for different targets
            ts, ps = ttest_1samp_masked(d, t0, mask=m, axis=axis)
            target_shape = list(d.shape)
            n = target_shape.pop(axis)
            assert_equal(ts.shape,  tuple(target_shape))

            def iterflat_view(a):
                return np.rollaxis(a, axis, 0).reshape((n, -1)).T

            # now compare to t-test with masking if done manually on
            for d_, m_, t_, p_ in zip(iterflat_view(d),
                                      iterflat_view(m),
                                      ts.flatten(),
                                      ps.flatten()):
                _assert_array_equal(ttest_1samp (d_[m_], t0), (t_, p_))

