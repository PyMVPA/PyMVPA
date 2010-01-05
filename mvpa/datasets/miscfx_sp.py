# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc function performing operations on datasets which are based on scipy
"""

__docformat__ = 'restructuredtext'

from mvpa.base import externals

import numpy as N

from operator import isSequenceType

from mvpa.base.dataset import datasetmethod
from mvpa.datasets.base import Dataset
from mvpa.misc.support import getBreakPoints

if externals.exists('scipy', raiseException=True):
    from scipy import signal
    from scipy.linalg import lstsq
    from scipy.special import legendre


@datasetmethod
def detrend(dataset, chunks=None, model='linear',
            polyord=None, opt_reg=None):
    """
    Given a dataset, detrend the data inplace either entirely
    or per each chunk

    :Parameters:
      dataset : Dataset
        dataset to operate on
      chunks : str, None
        If None, the whole dataset is detrended at once. Otherwise, the given
        samples attribute (given by its name) is used to define chunks of the
        dataset that are processed individually. In that case, all the samples
        within a chunk should be in contiguous order and the chunks should be
        sorted in order from low to high.
      model
        Type of detrending model to run.  If 'linear' or 'constant',
        scipy.signal.detrend is used to perform a linear or demeaning
        detrend. Polynomial detrending is activated when 'regress' is
        used or when polyord or opt_reg are specified.
      polyord : int or list
        Order of the Legendre polynomial to remove from the data.  This
        will remove every polynomial up to and including the provided
        value.  For example, 3 will remove 0th, 1st, 2nd, and 3rd order
        polynomials from the data.  N.B.: The 0th polynomial is the
        baseline shift, the 1st is the linear trend.
        If you specify a single int and `chunks` is not None, then this value
        is used for each chunk.  You can also specify a different polyord
        value for each chunk by providing a list or ndarray of polyord
        values the length of the number of chunks.
      opt_reg : ndarray
        Optional ndarray of additional information to regress out from the
        dataset.  One example would be to regress out motion parameters.
        As with the data, time is on the first axis.
    """
    if polyord is not None or opt_reg is not None:
        model='regress'

    if model in ['linear', 'constant']:
        # perform scipy detrend
        bp = 0                              # no break points by default

        if not chunks is None:
            try:
                bp = getBreakPoints(dataset.sa[chunks].value)
            except ValueError, e:
                raise ValueError, \
                      "Failed to assess break points between chunks. Often " \
                      "that is due to discontinuities within a chunk, which " \
                      "ruins idea of per-chunk detrending. Original " \
                      "exception was: %s" % str(e)

        dataset.samples[:] = signal.detrend(dataset.samples, axis=0,
                                            type=model, bp=bp)
    elif model in ['regress']:
        # perform regression-based detrend
        return __detrend_regress(dataset, chunks=chunks,
                                 polyord=polyord, opt_reg=opt_reg)
    else:
        # raise exception because not found
        raise ValueError('Specified model type (%s) is unknown.'
                         % (model))



def __detrend_regress(dataset, chunks, polyord, opt_reg):
    """The documentation is in the `detrend()` function."""
    #
    # Docum
    #

    # loop over chunks if necessary
    if not chunks is None:
        # get the unique chunks
        uchunks = dataset.sa[chunks].unique

        # Process the polyord to be a list with length of the number of chunks
        if not polyord is None:
            if not isSequenceType(polyord):
                # repeat to be proper length
                polyord = [polyord]*len(uchunks)
            elif not chunks is None and len(polyord) != len(uchunks):
                raise ValueError("If you specify a sequence of polyord values "
                                 "they sequence length must match the "
                                 "number of unique chunks in the dataset.")

        # loop over each chunk
        reg = []
        for n, chunk in enumerate(uchunks):
            # get the indices for that chunk
            cinds = dataset.sa[chunks].value == chunk

            # see if add in polyord values
            if not polyord is None:
                # create the timespan
                x = N.linspace(-1, 1, cinds.sum())
                # create each polyord with the value for that chunk
                for n in range(polyord[n] + 1):
                    newreg = N.zeros((len(dataset), 1))
                    newreg[cinds, 0] = legendre(n)(x)
                    reg.append(newreg)
    else:
        # take out mean over entire dataset
        reg = []
        # see if add in polyord values
        if not polyord is None:
            # create the timespan
            x = N.linspace(-1, 1, len(dataset))
            for n in range(polyord + 1):
                reg.append(legendre(n)(x)[:, N.newaxis])

    # see if add in optional regs
    if not opt_reg is None:
        # add in the optional regressors, too
        reg.append(opt_reg)

    # combine the regs
    if len(reg) > 0:
        if len(reg) > 1:
            regs = N.hstack(reg)
        else:
            regs = reg[0]
    else:
        # no regs to remove
        raise ValueError("You must specify at least one " + \
                             "regressor to regress out.")

    # perform the regression
    res = lstsq(regs, dataset.samples)

    # remove all but the residuals
    yhat = N.dot(regs, res[0])
    dataset.samples -= yhat

    # return the results
    return res, regs
