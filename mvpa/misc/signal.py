#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Simple preprocessing utilities"""

__docformat__ = 'restructuredtext'


import numpy as N
from scipy import signal
from scipy.linalg import lstsq
from scipy.special import legendre

from mvpa.misc.support import getBreakPoints

def detrend(data, perchunk=False, dtype='linear'):
    """
    Given a dataset, detrend the data inplace either entirely or per each chunk

    :Parameters:
      `data` : `Dataset`
        dataset to operate on
      `perchunk` : bool
        either to operate on whole dataset at once or on each chunk
        separately
      `dtype`
        type accepted by scipy.signal.detrend. Currently only
        'linear' or 'constant' (which is just demeaning)

    """

    bp = 0                              # no break points by default

    if perchunk:
        bp = getBreakPoints(data.chunks)

    data.samples[:] = signal.detrend(data.samples, axis=0, type=dtype, bp=bp)


def detrend_pattern(data, perchunk=True, polort=None, opt_reg=None):
    """
    Given a dataset, perform a detrend inplace, regressing out polynomial
    terms as well as optional regressors, such as motion parameters.

    :Parameters:
      `data`: `Dataset`
        Dataset to operate on
      `perchunk` : bool
        Either to operate on whole dataset at once or on each chunk
        separately.  If perchunk is True, all the samples within a
        chunk should be contiguous and the chunks should be sorted in
        order from low to high.
      `polort` : int
        Order of the Legendre polynomial to remove from the data.  This
        will remove every polynomial up to and including the provided
        value.  For example, 3 will remove 1st, 2nd, and 3rd order
        polynomials from the data.
      `opt_reg` : ndarray
        Optional ndarray of additional information to regress out from the
        dataset.  One example would be to regress out motion parameters.
        As with the data, time is on the first axis.
    """

    # create data to regress out
    # loop over chunks if necessary
    if perchunk:
        # get the unique chunks
        uchunks = data.uniquechunks

        # loop over each chunk
        cpol = []
        for chunk in uchunks:
            cinds = data.chunks == chunk
            x = N.linspace(-1, 1, cinds.sum())
            # create the polort for each chunk
            pol = []
            for n in range(1, polort + 1):
                pol.append(legendre(n)(x)[:, N.newaxis])
            cpol.append(N.hstack(pol))
        pols = N.vstack(cpol)
    else:
        # just create the polort over the entire dataset
        pol = []
        x = N.linspace(-1, 1, data.nsamples)
        for n in range(1, polort + 1):
            pol.append(legendre(n)(x)[:, N.newaxis])
        pols = N.hstack(pol)

    # combine all the regressors together
    tocombine = [N.ones((data.nsamples, 1))]
    if not polort is None:
        # add in the optional regressors, too
        tocombine.append(pols)
    if not opt_reg is None:
        # add in the optional regressors, too
        tocombine.append(opt_reg)
    regs = N.hstack(tocombine)

    # regress them out
    res = lstsq(regs, data.samples)

    # remove the residuals
    yhat = N.dot(regs, res[0])
    data.samples -= yhat

    # return the results
    return res, regs
