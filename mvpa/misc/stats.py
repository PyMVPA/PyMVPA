# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Little statistics helper"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals
externals.exists('scipy', raiseException=True)

import scipy.stats as stats
import numpy as N

def chisquare(obs, exp=None):
    """Compute the chisquare value of a contingency table with arbitrary
    dimensions.

    If no expected frequencies are supplied, the total N is assumed to be
    equally distributed across all cells.

    Returns: chisquare-stats, associated p-value (upper tail)
    """
    obs = N.array(obs)

    # get total number of observations
    nobs = N.sum(obs)

    # if no expected value are supplied assume equal distribution
    if exp == None:
        exp = N.ones(obs.shape) * nobs / N.prod(obs.shape)

    # make sure to have floating point data
    exp = exp.astype(float)

    # compute chisquare value
    chisq = N.sum((obs - exp )**2 / exp)

    # return chisq and probability (upper tail)
    return chisq, stats.chisqprob(chisq, N.prod(obs.shape) - 1)


