### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Little statistics helper
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the MIT License.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the COPYING
#    file that comes with this package for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

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


