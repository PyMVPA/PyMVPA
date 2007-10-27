#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA: Little statistics helper"""

import numpy as N
import scipy.stats as stats

def chisquare( obs, exp = None ):
    """ Compute the chisquare value of a contingency table with arbitrary
    dimensions.

    If no expected frequencies are supplied, the total N is assumed to be
    equally distributed across all cells.

    Returns: chisquare-stats, associated p-value (upper tail)
    """
    obs = N.array( obs )

    # get total number of observations
    N = N.sum( obs )

    # if no expected value are supplied assume equal distribution
    if exp == None:
        exp = N.ones(obs.shape) * N / N.prod(obs.shape)

    # make sure to have floating point data
    exp = exp.astype( float )

    # compute chisquare value
    chisq = N.sum( ( obs - exp )**2 / exp )

    # return chisq and probability (upper tail)
    return chisq, stats.chisqprob( chisq, N.prod( obs.shape ) - 1 )


