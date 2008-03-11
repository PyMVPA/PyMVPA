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

def detrend(data, perchunk=False, model='linear',
            polort=None, opt_reg=None):
    """
    Given a dataset, detrend the data inplace either entirely
    or per each chunk

    :Parameters:
      `data` : `Dataset`
        dataset to operate on
      `perchunk` : bool
        either to operate on whole dataset at once or on each chunk
        separately
      `model`
        Type of detrending model to run.  If 'linear' or 'constant',
        scipy.signal.detrend is used to perform a linear or demeaning
        detrend. If 'regress', then you specify the polort and opt_reg
        arguments to define regressors to regress out of the dataset.
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

    if model in ['linear', 'constant']:
        # perform scipy detrend
        bp = 0                              # no break points by default

        if perchunk:
            bp = getBreakPoints(data.chunks)

        data.samples[:] = signal.detrend(data.samples, axis=0,
                                         type=model, bp=bp)
    elif model in ['regress']:
        # perform regression-based detrend
        return __detrend_regress(data, perchunk=perchunk,
                                 polort=polort, opt_reg=opt_reg)
    else:
        # raise exception because not found
        raise ValueError('Specified model type (%s) is unknown.'
                         % (model))

def __detrend_regress(data, perchunk=True, polort=None, opt_reg=None):
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
    # Start a list of regressors, we always pull out mean
    regstocombine = [N.ones((data.nsamples, 1))]

    # loop over chunks if necessary
    if perchunk:
        # get the unique chunks
        uchunks = data.uniquechunks

        # loop over each chunk
        reg = []
        for chunk in uchunks:
            cinds = data.chunks == chunk
            # add in the baseline shift
            newreg = N.zeros((data.nsamples, 1))
            newreg[cinds,0] = N.ones(cinds.sum())
            reg.append(newreg)
            
            # see if add in polort values    
            if not polort is None:
                # create the timespan
                x = N.linspace(-1, 1, cinds.sum())
                # create each polort
                for n in range(1, polort + 1):
                    newreg = N.zeros((data.nsamples, 1))
                    newreg[cinds,0] = legendre(n)(x)
                    reg.append(newreg)
    else:
        # take out mean over entire dataset
        reg = [N.ones((data.nsamples, 1))]
        # see if add in polort values    
        if not polort is None:
            # create the timespan
            x = N.linspace(-1, 1, data.nsamples)
            for n in range(1, polort + 1):
                reg.append(legendre(n)(x)[:, N.newaxis])

    # see if add in optional regs
    if not opt_reg is None:
        # add in the optional regressors, too
        reg.append(opt_reg)

    # combine the regs
    if len(reg) > 1:
        regs = N.hstack(reg)
    else:
        regs = reg[0]

    # perform the regression
    res = lstsq(regs, data.samples)

    # remove all but the residuals
    yhat = N.dot(regs, res[0])
    data.samples -= yhat

    # return the results
    return res, regs
