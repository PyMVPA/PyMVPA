# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for dCOV and associated functions"""

from mvpa2.testing import *

# For testing
from nose.tools import ok_
from numpy.testing import assert_array_almost_equal
from mvpa2.testing.datasets import get_random_rotation

from mvpa2.misc.dcov import _euclidean_distances, dCOV, dcorcoef

from mvpa2.base import externals
if externals.exists('cran-energy'):
    from mvpa2.misc.dcov import dCOV_R


@reseed_rng()
def test_euclidean_distances():
    x = np.random.normal(size=(4, 10)) + np.random.normal() * 10
    d = _euclidean_distances(x, uv=True)
    # trust no one!
    distances = np.zeros((4, 10, 10))
    for ix, x_ in enumerate(x.T):
        for iy, y_ in enumerate(x.T):
            distances[:, ix, iy] = np.sqrt((x_ - y_) ** 2)
    assert_array_equal(d, distances)


def test_dCOV_against_R_energy():
    skip_if_no_external('cran-energy')

    for N in xrange(1, 10): # sweep through size of the first data
        # We will compare to R implementation
        M, T = 4, 30
        x = np.random.normal(size=(N, T)) + np.random.normal() * 10
        R = np.random.normal(size=(N, M))
        y = 10 * np.dot(R.T, x) + np.random.normal(size=(M, T)) \
            + np.random.normal(size=(M,))[:, None] # offset

        # To assure that works for not all_est
        pdCovs = dCOV(x, y, all_est=False)
        dCovs = dCOV_R(x, y, all_est=False)
        assert_array_almost_equal(pdCovs, dCovs)

        for uv in True, False:
            for out, outp in zip(dCOV_R(x, y, uv=uv),
                                 dCOV(x, y, uv=uv)):
                assert_array_almost_equal(out, outp)

@labile(5, 1)
def test_dCOV():
    # Few simple tests to verify that the measure seems to be ok
    for N in xrange(1, 10): # sweep through size of the first data
        # We will compare to R implementation
        M, T = 4, 100
        x = np.random.normal(size=(N, T)) + np.random.normal() * 10
        R = np.random.normal(size=(N, M))

        # linearly dependent variable after rotation
        dCov, dCor, _, _ = dCOV(x, 10 * np.dot(R.T, x))
        ok_(dCor > 0.7)           # should be really high but might fluctuate

        # completely independent variable
        dCov, dCor, _, _ = dCOV(x, np.random.normal(size=x.shape))
        # more dimension in x -- more uncertainty that they are
        # independent below is a heuristic (for T=100) and we should
        # just implement proper bootstrap significance estimation for
        # dCor
        ok_(dCor < 0.2 + N / 2.0)           # should be really high but might fluctuate

        # the same variable -- things should match for dCov and dVar's
        dCov, dCor, dVarx, dVary = dCOV(x, x)
        assert_equal(dCov, dVarx)
        assert_equal(dCov, dVary)
        assert_equal(dCor, 1.)
        assert_equal(dcorcoef(x, x), 1)
        #+ np.random.normal(size=(M, T)) \
        #    + np.random.normal(size=(M,))[:, None] # offset

        # Test that would work on vectors
        dCov, dCor, dVarx, dVary = dCOV(np.arange(N), np.sin(np.arange(N) / 3.))
        if N > 1:
            ok_(dCor > 0.6)           # should be really high but might fluctuate
        assert_equal(dcorcoef(np.arange(N), np.sin(np.arange(N) / 3.)), dCor)
