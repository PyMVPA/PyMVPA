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

from mvpa2.misc.fx import dual_gaussian, dual_positive_gaussian, fit2histogram

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
