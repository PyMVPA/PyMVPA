#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc function performing operations on datasets.
"""

__docformat__ = 'restructuredtext'

from sets import Set
from operator import isSequenceType

import numpy as N

from scipy import signal
from scipy.linalg import lstsq
from scipy.special import legendre

from mvpa.datasets import Dataset
from mvpa.misc.support import getBreakPoints



def zscore(dataset, mean=None, std=None,
           perchunk=True, baselinelabels=None,
           pervoxel=True, targetdtype='float64'):
    """Z-Score the samples of a `Dataset` (in-place).

    `mean` and `std` can be used to pass custom values to the z-scoring.
    Both may be scalars or arrays.

    All computations are done in place. Data upcasting is done
    automatically if necessary into `targetdtype`

    If `baselinelabels` provided, and `mean` or `std` aren't provided, it would
    compute the corresponding measure based only on labels in `baselinelabels`

    If `perchunk` is True samples within the same chunk are z-scored independent
    of samples from other chunks, e.i. mean and standard deviation are
    calculated individually.
    """
    # cast to floating point datatype if necessary
    if str(dataset.samples.dtype).startswith('uint') \
       or str(dataset.samples.dtype).startswith('int'):
        dataset.setSamplesDType(targetdtype)

    def doit(samples, mean, std, statsamples=None):
        """Internal method."""

        if statsamples is None:
            # if nothing provided  -- mean/std on all samples
            statsamples = samples

        if pervoxel:
            axisarg = {'axis':0}
        else:
            axisarg = {}

        # calculate mean if necessary
        if mean is None:
            mean = statsamples.mean(**axisarg)

        # de-mean
        samples -= mean

        # calculate std-deviation if necessary
        if std is None:
            std = statsamples.std(**axisarg)

        # do the z-scoring
        if pervoxel:
            samples[:, std != 0] /= std[std != 0]
        else:
            samples /= std

        return samples

    if baselinelabels is None:
        statids = None
    else:
        statids = Set(dataset.idsbylabels(baselinelabels))

    # for the sake of speed yoh didn't simply create a list
    # [True]*dataset.nsamples to provide easy selection of everything
    if perchunk:
        for c in dataset.uniquechunks:
            slicer = N.where(dataset.chunks == c)[0]
            if not statids is None:
                statslicer = list(statids.intersection(Set(slicer)))
                dataset.samples[slicer] = doit(dataset.samples[slicer],
                                               mean, std,
                                               dataset.samples[statslicer])
            else:
                slicedsamples = dataset.samples[slicer]
                dataset.samples[slicer] = doit(slicedsamples,
                                               mean, std,
                                               slicedsamples)
    elif statids is None:
        doit(dataset.samples, mean, std, dataset.samples)
    else:
        doit(dataset.samples, mean, std, dataset.samples[list(statids)])



def aggregateFeatures(dataset, fx):
    """Apply a function to each row of the samples matrix of a dataset.

    The functor given as `fx` has to honour an `axis` keyword argument in the
    way that NumPy used it (e.g. NumPy.mean, var).

    Returns a new `Dataset` object with the aggregated feature(s).
    """
    agg = fx(dataset.samples, axis=1)

    return Dataset(samples=N.array(agg, ndmin=2).T,
                   labels=dataset.labels,
                   chunks=dataset.chunks)



def removeInvariantFeatures(dataset):
    """Returns a new dataset with all invariant features removed.
    """
    return dataset.selectFeatures(dataset.samples.std(axis=0).nonzero()[0])



def detrend(dataset, perchunk=False, model='linear',
            polyord=None, opt_reg=None):
    """
    Given a dataset, detrend the data inplace either entirely
    or per each chunk

    :Parameters:
      `dataset` : `Dataset`
        dataset to operate on
      `perchunk` : bool
        either to operate on whole dataset at once or on each chunk
        separately
      `model`
        Type of detrending model to run.  If 'linear' or 'constant',
        scipy.signal.detrend is used to perform a linear or demeaning
        detrend. If 'regress', then you specify the polyord and opt_reg
        arguments to define regressors to regress out of the dataset.
      `polyord` : int or list
        Order of the Legendre polynomial to remove from the data.  This
        will remove every polynomial up to and including the provided
        value.  For example, 3 will remove 0th, 1st, 2nd, and 3rd order
        polynomials from the data.  N.B.: The 0th polynomial is the 
        baseline shift, the 1st is the linear trend.
        If you specify a single int and perchunk is True, then this value
        is used for each chunk.  You can also specify a differnt polyord 
        value for each chunk by providing a list or ndarray of polyord
        values the length of the number of chunks.
      `opt_reg` : ndarray
        Optional ndarray of additional information to regress out from the
        dataset.  One example would be to regress out motion parameters.
        As with the data, time is on the first axis.

    """

    if model in ['linear', 'constant']:
        # perform scipy detrend
        bp = 0                              # no break points by default

        if perchunk:
            bp = getBreakPoints(dataset.chunks)

        dataset.samples[:] = signal.detrend(dataset.samples, axis=0,
                                         type=model, bp=bp)
    elif model in ['regress']:
        # perform regression-based detrend
        return __detrend_regress(dataset, perchunk=perchunk,
                                 polyord=polyord, opt_reg=opt_reg)
    else:
        # raise exception because not found
        raise ValueError('Specified model type (%s) is unknown.'
                         % (model))



def __detrend_regress(dataset, perchunk=True, polyord=None, opt_reg=None):
    """
    Given a dataset, perform a detrend inplace, regressing out polynomial
    terms as well as optional regressors, such as motion parameters.

    :Parameters:
      `dataset`: `Dataset`
        Dataset to operate on
      `perchunk` : bool
        Either to operate on whole dataset at once or on each chunk
        separately.  If perchunk is True, all the samples within a
        chunk should be contiguous and the chunks should be sorted in
        order from low to high.
      `polyord` : int
        Order of the Legendre polynomial to remove from the data.  This
        will remove every polynomial up to and including the provided
        value.  For example, 3 will remove 0th, 1st, 2nd, and 3rd order
        polynomials from the data.  N.B.: The 0th polynomial is the 
        baseline shift, the 1st is the linear trend.
        If you specify a single int and perchunk is True, then this value
        is used for each chunk.  You can also specify a differnt polyord 
        value for each chunk by providing a list or ndarray of polyord
        values the length of the number of chunks.
      `opt_reg` : ndarray
        Optional ndarray of additional information to regress out from the
        dataset.  One example would be to regress out motion parameters.
        As with the data, time is on the first axis.
    """

    # Process the polyord to be a list with length of the number of chunks
    if not polyord is None:
        if not isSequenceType(polyord):
            # repeat to be proper length
            polyord = [polyord]*len(dataset.uniquechunks)
        elif perchunk and len(polyord) != len(dataset.uniquechunks):
            raise ValueError("If you specify a sequence of polyord values " + \
                                 "they sequence length must match the " + \
                                 "number of unique chunks in the dataset.")

    # loop over chunks if necessary
    if perchunk:
        # get the unique chunks
        uchunks = dataset.uniquechunks

        # loop over each chunk
        reg = []
        for n, chunk in enumerate(uchunks):
            # get the indices for that chunk
            cinds = dataset.chunks == chunk

            # see if add in polyord values    
            if not polyord is None:
                # create the timespan
                x = N.linspace(-1, 1, cinds.sum())
                # create each polyord with the value for that chunk
                for n in range(polyord[n] + 1):
                    newreg = N.zeros((dataset.nsamples, 1))
                    newreg[cinds, 0] = legendre(n)(x)
                    reg.append(newreg)
    else:
        # take out mean over entire dataset
        reg = []
        # see if add in polyord values    
        if not polyord is None:
            # create the timespan
            x = N.linspace(-1, 1, dataset.nsamples)
            for n in range(polyord[0] + 1):
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
