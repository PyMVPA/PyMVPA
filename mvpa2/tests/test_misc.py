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
    from mvpa2.misc.stats import *
    from scipy.stats import ttest_1samp
    from mvpa2.testing import *
    # test on some random data to match results of ttest_1samp
    d = np.random.normal(size=(5, 3))
    for null in 0, 0.5:
        # 1D case
        assert_array_equal(ttest_1samp       (d[0], null),
                           ttest_1samp_masked(d[0], null))

        for axis in 0, 1, None:
            assert_array_equal(ttest_1samp       (d, null, axis=axis),
                               ttest_1samp_masked(d, null, axis=axis))
    # we do not yet support >2D
    assert_raises(AssertionError, ttest_1samp_masked, d[None,...], 0)

    # basic test different tails
    d = range(10)
    tl, pl = ttest_1samp_masked(d, 0, tail='left')

    tr, pr = ttest_1samp_masked(d, 0, tail='right')
    tb, pb = ttest_1samp_masked(d, 0, tail='both')

    assert_equal(tl, tr)
    assert_equal(tl, tb)
    assert_equal(pl + pr, 1.0)
    assert_equal(pb, pl*2)
    assert(pl < 0.05)               # clearly we should be able to reject

    # finally let's get to masking
    # 1D
    d = np.arange(10)
    assert_array_equal(ttest_1samp       (d[3:], 0),
                       ttest_1samp_masked(d,     0,
                                          mask=[False]*3 + [True]*7))

    # random mask
    m = np.random.normal(size=d.shape) > 0.1
    assert_array_equal(ttest_1samp       (d[m], 0),
                       ttest_1samp_masked(d,    0, mask=m))

    # 2D masking
    d = np.arange(10).reshape((2,-1))
    m = np.random.normal(size=d.shape) > 0.1

    # axis=1
    ts, ps = ttest_1samp_masked(d, 0, mask=m, axis=1)
    for d_, m_, t_, p_ in zip(d, m, ts, ps):
        d_masked = d_[m_]
        assert_array_equal(ttest_1samp (d_[m_], 0), (t_, p_))

    # axis=0
    ts, ps = ttest_1samp_masked(d, 0, mask=m, axis=0)
    for d_, m_, t_, p_ in zip(d.T, m.T, ts, ps):
        d_masked = d_[m_]
        assert_array_equal(ttest_1samp (d_[m_], 0), (t_, p_))

