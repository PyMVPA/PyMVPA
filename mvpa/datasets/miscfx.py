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

from mvpa.datasets import Dataset
from mvpa.misc.support import getBreakPoints

from mvpa.base import externals

if externals.exists('scipy'):
    from mvpa.datasets.miscfx_sp import detrend

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

